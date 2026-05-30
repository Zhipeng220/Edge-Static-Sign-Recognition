from pathlib import Path
import sys

_UTILS_ORIGINAL = Path(__file__).resolve().parents[1] / "utils_original"
if str(_UTILS_ORIGINAL) not in sys.path:
    sys.path.insert(0, str(_UTILS_ORIGINAL))

from gesture_experiment.data import (  # noqa: F401
    canonicalize_label,
    canonicalize_labels,
    labels_to_names,
    normalize_single_hand_landmarks,
    to_single_hand_features,
)
