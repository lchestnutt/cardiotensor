import sys
import os
from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import dask
import numpy as np
import psutil
import SimpleITK as sitk
from alive_progress import alive_bar
from scipy.ndimage import zoom
from skimage.measure import block_reduce
import tifffile as tiff

from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------
# Integer-only upsampling util
# ---------------------------
def _upsample_mask_integer(mask: np.ndarray, zf: int, yf: int, xf: int) -> np.ndarray:
    """Pure-NumPy nearest-neighbor upsampling for integer factors."""
    out = mask
    if zf != 1:
        out = np.repeat(out, zf, axis=0)
    if yf != 1:
        out = np.repeat(out, yf, axis=1)
    if xf != 1:
        out = np.repeat(out, xf, axis=2)
    return out

def _fit(arr: np.ndarray, target: tuple[int, int, int], pad_value: int = 0) -> np.ndarray:
    """
    Force `arr` (Z, Y, X) to exactly match `target` shape.
    Crops if too large, pads with `pad_value` if too small.
    """
    tz, ty, tx = target
    arr = arr[:tz, :ty, :tx]
    padz = max(0, tz - arr.shape[0])
    pady = max(0, ty - arr.shape[1])
    padx = max(0, tx - arr.shape[2])
    if padz or pady or padx:
        arr = np.pad(arr,
                     ((0, padz), (0, pady), (0, padx)),
                     mode="constant",
                     constant_values=pad_value)
    return arr



class DataReader:
    def __init__(self, path: str | Path):
        """
        Initializes the DataReader with a path to the volume.

        Args:
            path (str | Path): Path to the volume directory or file.
        """
        self.path = Path(path)
        self.supported_extensions = ["tif", "tiff", "jp2", "png", "npy"]
        self.volume_info = self._get_volume_info()

    # ---------------------------
    # Properties
    # ---------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the volume as (Z, Y, X) or (Z, Y, X, C)."""
        return self.volume_info["shape"]

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the volume."""
        return self.volume_info["dtype"]

    @property
    def volume_size_gb(self) -> float:
        """Returns the total size of the volume in GB."""
        n_bytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize
        return n_bytes / (1024**3)

    def _get_volume_info(self) -> dict:
        """
        Detects volume type, shape, and dtype.
        Returns a dict with keys: type, stack, file_list, shape, dtype
        """
        volume_info = {
            "type": "",
            "stack": False,
            "file_list": [],
            "shape": None,
            "dtype": None,
        }

        if not self.path.exists():
            raise ValueError(f"The path does not exist: {self.path}")

        # Case 1: Directory of images
        if self.path.is_dir():
            volume_info["stack"] = True
            image_files = {
                ext: sorted(self.path.glob(f"*.{ext}"))
                for ext in self.supported_extensions
            }
            volume_info["type"], volume_info["file_list"] = max(
                image_files.items(), key=lambda item: len(item[1])
            )
            if not volume_info["file_list"]:
                raise ValueError(
                    "No supported image files found in the specified directory."
                )

            # Inspect first file
            first_image = self._custom_image_reader(volume_info["file_list"][0])           
            volume_info["dtype"] = first_image.dtype

            # Shape: handle scalar vs vector
            if (
                volume_info["type"] == "npy"
                and first_image.ndim == 3
                and first_image.shape[0] == 3
            ):
                # Vector field stored as (3, Y, X)
                volume_info["shape"] = (
                    3,
                    len(volume_info["file_list"]),
                    first_image.shape[1],
                    first_image.shape[2],
                )
            elif first_image.ndim == 3:
                # 4D scalar stack (Z from files)
                volume_info["shape"] = (
                    len(volume_info["file_list"]),
                    *first_image.shape,
                )
            else:
                # Standard 3D stack
                volume_info["shape"] = (
                    len(volume_info["file_list"]),
                    first_image.shape[0],
                    first_image.shape[1],
                )

        # Case 2: Single MHD file
        elif self.path.is_file() and self.path.suffix == ".mhd":
            volume_info["type"] = "mhd"
            img = sitk.ReadImage(str(self.path))
            arr = sitk.GetArrayFromImage(img)  # Z, Y, X
            volume_info["shape"] = arr.shape
            volume_info["dtype"] = arr.dtype

        else:
            raise ValueError(f"Unsupported volume type for path: {self.path}")

        return volume_info

    def load_volume(
        self,
        start_index: int = 0,
        end_index: int | None = None,
        unbinned_shape: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """
        Loads the volume and resizes it to unbinned_shape if provided, using fast
        integer-only resampling:
        - np.repeat for upsampling
        - block_reduce (max) for downsampling

        Args:
            start_index (int): Start index for slicing (for stacks).
            end_index (int): End index for slicing (for stacks). If None, loads the entire stack.
            unbinned_shape (tuple): Desired shape (Z, Y, X). If None, no resizing is done.

        Returns:
            np.ndarray: Loaded volume.
        """
        if end_index is None:
            end_index = self.shape[0]

        # Check memory available is enough
        effective_shape = list(self.shape)
        if len(effective_shape) == 3:
            effective_shape[0] = end_index - start_index
        elif len(effective_shape) == 4:
            effective_shape[1] = end_index - start_index
        self.check_memory_requirement(tuple(effective_shape), self.dtype)

        # Decide if resize is needed
        need_resize = False
        if unbinned_shape is not None and self.shape != unbinned_shape:
            need_resize = True
            zoom_factors = tuple(u / s for u, s in zip(unbinned_shape, self.shape))
            print(f"Resample factors: {zoom_factors}")
        else:
            zoom_factors = (1.0, 1.0, 1.0)

        if need_resize:
            start_index_ini, end_index_ini = start_index, end_index
            start_index = int(start_index_ini / zoom_factors[0]) - 1
            start_index = max(start_index, 0)
            end_index = int(end_index_ini / zoom_factors[0]) + 1
            end_index = min(end_index, self.shape[0])
            print(f"Volume start index padded: {start_index} - end: {end_index}")

        # Load volume from stack or mhd
        if not self.volume_info["stack"]:
            if self.volume_info["type"] == "mhd":
                volume, _ = _load_raw_data_with_mhd(self.path)
                volume = volume[start_index:end_index, :, :]
            else:
                raise ValueError(f"Unsupported volume type for path: {self.path}")
        else:
            volume = self._load_image_stack(
                self.volume_info["file_list"], start_index, end_index
            )

        if need_resize:
            print("Resizing with integer-only resampling...")

            _, y1, x1 = unbinned_shape
            z1 = end_index_ini - start_index_ini
            fz, fy, fx = zoom_factors

            def _as_int_factor_up(f: float) -> int | None:
                k = int(round(f))
                return k if (k >= 1 and abs(f - k) < 1e-1) else None

            def _as_int_factor_down(f: float) -> int | None:
                # f < 1 expected, so divisor ~ round(1/f)
                if f >= 1:
                    return None
                k = int(round(1.0 / f))
                return k if k >= 1 and abs(f - (1.0 / k)) < 1e-1 else None

            # Z
            if fz > 1:
                kz = _as_int_factor_up(fz)
                if kz is None:
                    raise ValueError(f"Non-integer upsample factor on Z: {fz}")
                volume = np.repeat(volume, kz, axis=0)
            elif fz < 1:
                dz = _as_int_factor_down(fz)
                if dz is None:
                    raise ValueError(f"Non-integer downsample factor on Z: {fz}")
                volume = block_reduce(volume, block_size=(dz, 1, 1), func=np.max)

            # Y
            if fy > 1:
                ky = _as_int_factor_up(fy)
                if ky is None:
                    raise ValueError(f"Non-integer upsample factor on Y: {fy}")
                volume = np.repeat(volume, ky, axis=1)
            elif fy < 1:
                dy = _as_int_factor_down(fy)
                if dy is None:
                    raise ValueError(f"Non-integer downsample factor on Y: {fy}")
                volume = block_reduce(volume, block_size=(1, dy, 1), func=np.max)

            # X
            if fx > 1:
                kx = _as_int_factor_up(fx)
                if kx is None:
                    raise ValueError(f"Non-integer upsample factor on X: {fx}")
                volume = np.repeat(volume, kx, axis=2)
            elif fx < 1:
                dx = _as_int_factor_down(fx)
                if dx is None:
                    raise ValueError(f"Non-integer downsample factor on X: {fx}")
                volume = block_reduce(volume, block_size=(1, 1, dx), func=np.max)
            
            print(f"Resample factors: [{kz}, {ky}, {kx}]")

            # Enforce exact shape 
            volume = _fit(volume, (z1, y1, x1), pad_value=0)

        return volume


    def _custom_image_reader(self, file_path: Path) -> np.ndarray:
        """
        Reads an image from the given file path into a NumPy array.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            np.ndarray
        """
        suffix = file_path.suffix.lower()

        # 1) .npy → memory-mapped (no full copy)
        if suffix == ".npy":
            return np.load(file_path, mmap_mode="r")

        # 2) TIFF → use tifffile (OpenCV often fails on SampleFormat/BigTIFF)
        if suffix in {".tif", ".tiff"}:
            try:
                # imread handles most TIFF flavors; memmap if you want zero-copy:
                # arr = tiff.memmap(file_path)  # if you prefer memmap
                arr = tiff.imread(file_path)
                # Optional: normalize binary masks to uint8 if values are 0/1
                if arr.dtype.kind in ("i", "u") and arr.max() <= 1:
                    arr = arr.astype(np.uint8)
                return arr
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read TIFF '{file_path}' with tifffile: {e}"
                ) from e

        # 3) Other formats → OpenCV (allow any depth/color)
        img = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise RuntimeError(
                f"cv2.imread failed for '{file_path}'. "
                "File may be missing or in an unsupported/invalid format."
            )
        return img


    def _load_image_stack(
        self, file_list: list[Path], start_index: int, end_index: int
    ) -> np.ndarray:
        """
        Loads a stack of images into a 3D NumPy array.

        Args:
            file_list (List[Path]): List of file paths to load.
            start_index (int): Start index for slicing.
            end_index (int): End index for slicing.

        Returns:
            np.ndarray: Loaded volume as a 3D array.
        """
        if end_index == 0:
            end_index = len(file_list)

        total_files = end_index - start_index
        # print(f"Loading {total_files} files...")

        if start_index < 0 or end_index > len(file_list):
            raise ValueError(
                f"Invalid indices: start_index={start_index}, end_index={end_index}, total_files={len(file_list)}"
            )

        with alive_bar(total_files, title="Loading Volume", length=40) as bar:

            def progress_bar_reader(file_path: Path) -> np.ndarray:
                bar()  # Update the progress bar
                return self._custom_image_reader(file_path)

            delayed_tasks = [
                dask.delayed(progress_bar_reader)(file_path)
                for file_path in sorted(file_list[start_index:end_index])
            ]

            # Compute the volume
            computed_data = dask.compute(*delayed_tasks, scheduler="threads")

            # Validate shape consistency
            first_shape = computed_data[0].shape
            for idx, data in enumerate(computed_data):
                if data.shape != first_shape:
                    raise ValueError(
                        f"Inconsistent file shape at index {idx}: Expected {first_shape}, got {data.shape}"
                    )

        # Combine into a NumPy array
        print("Stacking images into a 3D volume...")
        if self.volume_info["type"] == "npy":
            volume = np.stack(computed_data, axis=1)
        else:
            volume = np.stack(computed_data, axis=0)

        return volume

    def check_memory_requirement(self, shape, dtype, safety_factor=0.8):
        """
        Check if the dataset can fit in available memory.

        Args:
            shape (tuple[int]): Shape of the array.
            dtype (np.dtype): NumPy dtype of the array.
            safety_factor (float): Fraction of available memory allowed to be used.
        """
        # Compute dataset size in bytes
        n_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        size_gb = n_bytes / (1024**3)

        # Check available memory
        available_gb = psutil.virtual_memory().available / (1024**3)

        print(
            f"Dataset size: {size_gb:.2f} GB | Available memory: {available_gb:.2f} GB"
        )

        if size_gb > available_gb * safety_factor:
            print("❌ Dataset is too large to safely load into memory.")
            sys.exit(1)


def _read_mhd(filename: PathLike[str]) -> dict[str, Any]:
    """
    Return a dictionary of meta data from an MHD meta header file.

    Args:
        filename (PathLike[str]): File path to the .mhd file.

    Returns:
        dict[str, Any]: A dictionary containing parsed metadata.
    """
    meta_dict: dict[str, Any] = {}
    tag_set = [
        "ObjectType",
        "NDims",
        "DimSize",
        "ElementType",
        "ElementDataFile",
        "ElementNumberOfChannels",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "CompressedData",
        "CompressedDataSize",
        "Offset",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "TransformMatrix",
        "Comment",
        "SeriesDescription",
        "AcquisitionDate",
        "AcquisitionTime",
        "StudyDate",
        "StudyTime",
    ]

    with open(filename) as fn:
        for line in fn:
            tags = line.split("=")
            if len(tags) < 2:
                continue
            key, content = tags[0].strip(), tags[1].strip()
            if key in tag_set:
                if key in [
                    "ElementSpacing",
                    "Offset",
                    "CenterOfRotation",
                    "TransformMatrix",
                ]:
                    # Parse as a list of floats
                    meta_dict[key] = [float(value) for value in content.split()]
                elif key in ["NDims", "ElementNumberOfChannels"]:
                    # Parse as an integer
                    meta_dict[key] = int(content)
                elif key == "DimSize":
                    # Parse as a list of integers
                    meta_dict[key] = [int(value) for value in content.split()]
                elif key in ["BinaryData", "BinaryDataByteOrderMSB", "CompressedData"]:
                    # Parse as a boolean
                    meta_dict[key] = content.lower() == "true"
                else:
                    # Parse as a string
                    meta_dict[key] = content
    return meta_dict


def _load_raw_data_with_mhd(
    filename: PathLike[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a MHD file

    :param filename: file of type .mhd that should be loaded
    :returns: tuple with raw data and dictionary of meta data
    """
    meta_dict = _read_mhd(filename)
    dim = int(meta_dict["NDims"])
    if "ElementNumberOfChannels" in meta_dict:
        element_channels = int(meta_dict["ElementNumberOfChannels"])
    else:
        element_channels = 1

    if meta_dict["ElementType"] == "MET_FLOAT":
        np_type = np.float32
    elif meta_dict["ElementType"] == "MET_DOUBLE":
        np_type = np.float64
    elif meta_dict["ElementType"] == "MET_CHAR":
        np_type = np.byte
    elif meta_dict["ElementType"] == "MET_UCHAR":
        np_type = np.ubyte
    elif meta_dict["ElementType"] == "MET_SHORT":
        np_type = np.int16
    elif meta_dict["ElementType"] == "MET_USHORT":
        np_type = np.ushort
    elif meta_dict["ElementType"] == "MET_INT":
        np_type = np.int32
    elif meta_dict["ElementType"] == "MET_UINT":
        np_type = np.uint32
    else:
        raise NotImplementedError(
            "ElementType " + meta_dict["ElementType"] + " not understood."
        )
    arr = list(meta_dict["DimSize"])

    volume = np.prod(arr[0 : dim - 1])

    pwd = Path(filename).parents[0].resolve()
    data_file = Path(meta_dict["ElementDataFile"])
    if not data_file.is_absolute():
        data_file = pwd / data_file

    shape = (arr[dim - 1], volume, element_channels)
    with open(data_file, "rb") as f:
        data = np.fromfile(f, count=np.prod(shape), dtype=np_type)
    data.shape = shape

    # Adjust byte order in numpy array to match default system byte order
    if "BinaryDataByteOrderMSB" in meta_dict:
        sys_byteorder_msb = sys.byteorder == "big"
        file_byteorder_ms = meta_dict["BinaryDataByteOrderMSB"]
        if sys_byteorder_msb != file_byteorder_ms:
            data = data.byteswap()

    # Begin 3D fix
    # arr.reverse()
    if element_channels > 1:
        data = data.reshape(arr + [element_channels])
    else:
        data = data.reshape(arr)
    # End 3D fix

    return (data, meta_dict)
