from pathlib import Path
import numpy as np
from numpy import ndarray
from io import BytesIO
from upath import UPath


def resolved_path(p: str | UPath) -> Path:
    return UPath(p).expanduser().resolve()


def load(p: str | UPath) -> ndarray:
    # TODO: choose the appropriate function to load data by p.suffix
    ...


def ndarray2bytes(arr: ndarray) -> bytes:
    with BytesIO() as buffer:
        np.save(buffer, arr)
        out = buffer.getvalue()
    return out


def all_exist(*paths: UPath) -> bool:
    for path in paths:
        if not path.exists():
            return False


PathOrNone = Path | None
