import glob
import inspect
import math
import os
import subprocess
import sys
import time
from datetime import datetime

from cardiotensor.orientation.orientation_computation_pipeline import (
    compute_orientation,
)
from cardiotensor.utils.DataReader import DataReader
from cardiotensor.utils.utils import (
    read_conf_file,
)


def submit_job_to_slurm(
    executable_path: str,
    conf_file_path: str,
    start_image: int,
    end_image: int,
    N_chunk: int = 10,
    mem_needed: int = 64,
) -> int:
    """
    Submit a Slurm job and return its job ID.

    Args:
        executable_path (str): Path to the executable script.
        conf_file_path (str): Path to the configuration file.
        start_image (int): Index of the first image to process.
        end_image (int): Index of the last image to process.
        N_chunk (int, optional): Number of chunks for the job. Default is 10.
        mem_needed (int, optional): Memory required in GB. Default is 64.

    Returns:
        int: The Slurm job ID.
    """
    log_dir = "/tmp_14_days/bm18/slurm/log/"
    submit_dir = "/tmp_14_days/bm18/slurm/submit"

    executable_path = executable_path.split(".py")[0]
    executable = os.path.basename(executable_path)
    print(f"Script to start: {executable}", flush=True)

    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")

    job_name = f"{executable}_{current_date}"
    # Generate the job filename
    job_filename = f"{submit_dir}/{job_name}.slurm"

    # Calculate the total number of images to process
    total_images = end_image - start_image + 1
    IMAGES_PER_JOB = N_chunk
    N_jobs = math.ceil(total_images / IMAGES_PER_JOB)

    print(f"\nN_jobs = {N_jobs}, IMAGES_PER_JOB = {IMAGES_PER_JOB}", flush=True)

    slurm_script_content = f"""#!/bin/bash -l
#SBATCH --output={log_dir}/slurm-%x-%A_%a.out
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={math.ceil(mem_needed)}G
#SBATCH --job-name={job_name}
#SBATCH --time=2:00:00
#SBATCH --array=1-{N_jobs}%50

echo ------------------------------------------------------
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST
echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR
echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo SLURM_JOB_NAME: $SLURM_JOB_NAME
echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION
echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_CPUS-PER-TASK: $SLURM_CPUS_PER_TASK
echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU
echo SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE
echo ------------------------------------------------------


IMAGES_PER_JOB={IMAGES_PER_JOB}
START_IMAGE={start_image}
TOTAL_IMAGES={total_images + start_image}

START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * $IMAGES_PER_JOB + $START_IMAGE ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * $IMAGES_PER_JOB + $START_IMAGE ))

# Cap END_INDEX to the last image index (zero-based)
if [ $END_INDEX -ge $TOTAL_IMAGES ]; then END_INDEX=$(( $TOTAL_IMAGES + $START_IMAGE - 1 )); fi

echo Start index, End index : $START_INDEX: $END_INDEX
echo mem used {math.ceil(mem_needed)}G

# Fix Qt headless error
export QT_QPA_PLATFORM=offscreen

# Starting python script
echo cardio-tensor {conf_file_path} --start_index $START_INDEX --end_index $END_INDEX

# cardio-tensor {executable_path}.py {conf_file_path} --start_index $START_INDEX --end_index $END_INDEX
cardio-tensor {conf_file_path} --start_index $START_INDEX --end_index $END_INDEX

"""

    with open(job_filename, "w") as file:
        file.write(slurm_script_content)

    try:
        result = subprocess.run(
            ["sbatch", job_filename], capture_output=True, text=True, check=True
        )
        job_id = result.stdout.split()[-1]
        print(f"sbatch {job_id} - Index {start_image} to {end_image}", flush=True)
        return int(job_id)
    except subprocess.CalledProcessError:
        print(f"⚠️ - Failed to submit Slurm job with script {job_filename}", flush=True)
        sys.exit()



def slurm_launcher(conf_file_path: str, start_index: int = 0, end_index: int | None = None) -> None:
    """
    Launch Slurm array jobs for a subset [start_index, end_index] (inclusive) of the volume.
    If end_index is None, the last slice of the volume is used.
    """
    try:
        params = read_conf_file(conf_file_path)
    except Exception as e:
        print(f"⚠️  Error reading parameter file '{conf_file_path}': {e}", flush=True)
        sys.exit(1)

    VOLUME_PATH = params.get("IMAGES_PATH", "")
    OUTPUT_DIR = params.get("OUTPUT_PATH", "./output")
    N_CHUNK = int(params.get("N_CHUNK", 100))
    IS_TEST = params.get("TEST", False)

    if IS_TEST is True:
        sys.exit(
            "Test mode activated, run directly 3D_processing.py or deactivate test mode in the parameter file"
        )

    data_reader = DataReader(VOLUME_PATH)
    total_slices = int(data_reader.shape[0])

    # ----- sanitize window -----
    first = max(0, int(start_index))
    last = total_slices - 1 if end_index is None else int(end_index)
    if last < 0:
        last = total_slices - 1

    # clamp
    first = max(0, min(first, total_slices - 1))
    last = max(0, min(last, total_slices - 1))

    if last < first:
        print(f"⚠️ Invalid range: start_index ({first}) > end_index ({last})", flush=True)
        sys.exit(1)

    window_len = last - first + 1
    print(f"Processing slice window [{first}, {last}] (len={window_len}) out of 0..{total_slices-1}", flush=True)

    mem_needed = 128

    # ----- build per-job intervals (inclusive) -----
    def build_intervals(first_idx: int, last_idx: int, step: int) -> list[tuple[int, int]]:
        out = []
        s = first_idx
        while s <= last_idx:
            e = min(s + step - 1, last_idx)
            out.append((s, e))
            s = e + 1
        return out

    intervals = build_intervals(first, last, N_CHUNK)
    n_jobs_total = len(intervals)
    print(f"Splitting data into {n_jobs_total} jobs of up to {N_CHUNK} slices each", flush=True)

    # ----- batch into arrays of <= 999 tasks -----
    N_job_max_per_array = 999
    batched = [intervals[i:i + N_job_max_per_array] for i in range(0, n_jobs_total, N_job_max_per_array)]
    print(
        f"Launching {len(batched)} arrays (tasks per array: {[len(b) for b in batched]})",
        flush=True,
    )

    python_file_path = os.path.abspath(inspect.getfile(compute_orientation))
    start_t = time.time()

    for batch in batched:
        batch_start = batch[0][0]
        batch_end = batch[-1][1]  # inclusive
        job_id = submit_job_to_slurm(
            python_file_path,
            conf_file_path,
            batch_start,
            batch_end,
            N_chunk=N_CHUNK,
            mem_needed=mem_needed,
        )
        print(
            f"Submitted array for [{batch_start}, {batch_end}] (job ID: {job_id})",
            flush=True,
        )

    # monitor only the requested window
    monitor_job_output(OUTPUT_DIR, window_len, conf_file_path)

    print(f"Execution seconds: {time.time() - start_t}", flush=True)


def monitor_job_output(
    output_directory: str, total_images: int, file_extension: str
) -> None:
    """
    Monitor OUTPUT_DIR/HA until total_images files appear (subset-aware).
    """
    start_time = time.time()
    time.sleep(1)
    tmp_count = len(glob.glob(f"{output_directory}/HA/*"))
    while True:
        current_files_count = len(glob.glob(f"{output_directory}/HA/*"))

        processed = current_files_count
        print(f"{processed}/{total_images} processed", flush=True)

        if current_files_count > tmp_count:
            rate = current_files_count - tmp_count  # images per minute (since we sleep 60s)
            remaining = max(total_images - processed, 0)
            eta_min = remaining / rate if rate > 0 else float("inf")
            print(
                f"{current_files_count - tmp_count} images processed in 60sec. "
                f"Approximately {eta_min:.2f} minutes remaining",
                flush=True,
            )
        tmp_count = current_files_count

        if processed >= total_images:
            break

        print(f"Processing time (s): {time.time() - start_time:.1f}", flush=True)
        print("\nWaiting 60 seconds...\n", flush=True)
        time.sleep(60)


def is_chunk_done(
    output_dir: str, start: int, end: int, output_format: str = "jp2"
) -> bool:
    """
    Check if all output files (HA, IA, FA) for a given chunk are already present.

    Args:
        output_dir (str): Base output directory containing HA/IA/FA folders.
        start (int): Start index of the chunk (inclusive).
        end (int): End index of the chunk (exclusive).
        output_format (str): File extension for the output images (e.g., "jp2", "tif").

    Returns:
        bool: True if all expected output files exist, False otherwise.
    """
    for idx in range(start, end):
        expected_files = [
            f"{output_dir}/HA/HA_{idx:06d}.{output_format}",
            f"{output_dir}/IA/IA_{idx:06d}.{output_format}",
            f"{output_dir}/FA/FA_{idx:06d}.{output_format}",
        ]
        if not all(os.path.exists(f) for f in expected_files):
            return False
    return True
