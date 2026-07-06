import os
from typing import Optional, Tuple

import numpy as np


def atomic_save_npy(path: str, array: np.ndarray) -> None:
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def load_npy_if_ready(path: str, expected_shape: Optional[Tuple[int, ...]] = None):
    try:
        data = np.load(path)
    except (FileNotFoundError, ValueError, EOFError, OSError):
        return None
    if expected_shape is not None and data.shape != expected_shape:
        return None
    return data
