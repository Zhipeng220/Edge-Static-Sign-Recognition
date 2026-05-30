from pathlib import Path
import sys


def add_original_utils_to_path() -> None:
    """Allow direct script execution without requiring PYTHONPATH."""
    utils_path = Path(__file__).resolve().parent / "utils_original"
    utils_path_str = str(utils_path)
    if utils_path_str not in sys.path:
        sys.path.insert(0, utils_path_str)
