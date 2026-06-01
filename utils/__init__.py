from pathlib import Path
import numpy as np
from numpy import ndarray
from io import BytesIO
from upath import UPath
import nibabel
from errors import *


def resolved_path(p: str | UPath) -> Path:
    return UPath(p).expanduser().resolve()


def _list_endswith(f: list, e: list) -> bool:
    return f[-len(e) :] == e


def is_nii(p: UPath | str | None):
    if not p:
        return None
    return _list_endswith(p.suffixes, [".nii", ".gz"]) or _list_endswith(
        p.suffix, [".nii"]
    )


def dice(a: np.ndarray, b: np.ndarray, l: int):
    return (2 * np.count_nonzero((a == l) == (b == l))) / (
        np.count_nonzero(a == l) + np.count_nonzero(b == l)
    )


def load(p: str | UPath) -> ndarray:
    """dim=3: channel, saggital, coronal, axial"""
    p = resolved_path(p)
    if is_nii(p):
        img = nibabel.load(p)
        loaded: ndarray = nibabel.as_closest_canonical(img).get_fdata()
        if len(loaded.shape) == 4:
            loaded = np.transpose(loaded, [3, 0, 1, 2])
        return loaded
    else:
        raise InternalError("unimplemented load suffix", payload={"path": p})


def ndarray2bytes(arr: ndarray) -> bytes:
    with BytesIO() as buffer:
        np.save(buffer, arr)
        out = buffer.getvalue()
    return out


def all_exist(*paths: UPath) -> bool:
    for path in paths:
        if not path.exists():
            return False


def map_dict(mapper, dic: dict):
    return {k: mapper(v) for k, v in dic.items()}
