from pathlib import Path
import numpy as np
from numpy import ndarray
from io import BytesIO
from enum import Enum, auto


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


def all_exist(*paths: Path) -> bool:
    for path in paths:
        if not path.exists():
            return False
    return True


PathOrNone = Path | None
