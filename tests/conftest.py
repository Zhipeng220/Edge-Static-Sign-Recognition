from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ORIGINAL_UTILS_DIR = SRC_DIR / "utils_original"

for path in (SRC_DIR, ORIGINAL_UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
