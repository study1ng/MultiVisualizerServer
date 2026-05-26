from pathlib import Path
import numpy as np
from numpy import ndarray
from io import BytesIO


def resolved_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def load(p: str | Path) -> ndarray:
    # TODO: choose the appropriate function to load data by p.suffix
    ...


def ndarray2bytes(arr: ndarray) -> bytes:
    with BytesIO() as buffer:
        np.save(buffer, arr)
        out = buffer.getvalue()
    return out
