"""
3D_Data_Processing
"""

import argparse
import os
import sys

from cardiotensor.launcher.slurm_launcher import slurm_launcher


def script() -> None:
    """
    Main script to process 3D data. Reads configuration files, launches processing tasks,
    and logs processing time.
    """

    # Parse the configuration file path from command-line arguments
    parser = argparse.ArgumentParser(
        description="Process 3D data using the specified configuration file."
    )
    parser.add_argument(
        "conf_file_path", type=str, help="Path to the input configuration file."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting slice index for processing (default: 0).",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Ending slice index (default: None, processes all slices).",
    )
    args = parser.parse_args()
    conf_file_path = args.conf_file_path

    # Launch processing using slurm_launcher
    slurm_launcher(conf_file_path, start_index=args.start_index, end_index=args.end_index)


if __name__ == "__main__":
    script()
