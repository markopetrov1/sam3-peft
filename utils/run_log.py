"""
Timestamped run directories and full run logging (stdout tee + optional logging).

Use from train.py and test.py so every print() and logging call is captured
in a single log file from the start of the run.
"""

import logging
import os
import sys
from datetime import datetime


def timestamp() -> str:
    """Current time as YYYY-MM-DD_HH-MM-SS for use in directory names."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Tee:
    """
    Writes to both original stdout and a log file so all print() output
    is captured in the run log.
    """

    def __init__(self, stdout, log_file):
        self._stdout = stdout
        self._log = log_file

    def write(self, s):
        self._stdout.write(s)
        self._log.write(s)
        self._log.flush()

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def isatty(self):
        return self._stdout.isatty()


def setup_run_log(
    parent_dir: str,
    dir_name: str,
    log_filename: str = "run.log",
    use_logging: bool = True,
) -> str:
    """
    Create a run directory, tee stdout to a log file, and optionally
    configure logging to that file and terminal.

    Args:
        parent_dir: Base directory (e.g. experiments).
        dir_name: Subdirectory name (e.g. sam3_lora_potsdam_2026-02-22_10-52-00).
        log_filename: Name of the log file inside the run dir.
        use_logging: If True, configure the root logger to write to the log
            file and original stdout with [HH:MM:SS] prefix.

    Returns:
        Absolute path to the run directory (parent_dir/dir_name).
    """
    run_dir = os.path.join(parent_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_filename)
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_file)

    if use_logging:
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        log.handlers.clear()
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        for handler in (logging.StreamHandler(log_file), logging.StreamHandler(orig_stdout)):
            handler.setFormatter(fmt)
            log.addHandler(handler)

    return run_dir
