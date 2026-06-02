from errors import ServerError
from methods_types import Method
from utils import load
from upath import UPath
import numpy as np
from errors import *


class Normal(Method):
    """just load all and return"""

    def __init__(
        self,
    ):
        pass

    @staticmethod
    def _load(p):
        if p is None:
            return None

        if not isinstance(p, (UPath, str)):
            raise ServerError("ServerError: p should be Path or str")

        return load(p)

    def _process(self, base, gt, fn):
        ret = {}
        bl = self._load(base)
        gl = self._load(gt)
        fl = [self._load(f) for f in fn]

        def can_cast(a: np.ndarray, d):
            return np.allclose(a.astype(d), a)

        if gl is not None and not can_cast(gl, np.uint32):
            raise InternalError(
                "a label dtype is not castable to uint32",
                payload={"dt": gl},
            )

        for f in fl:
            if not can_cast(f, np.uint32):
                raise InternalError(
                    "a label dtype is not castable to uint32",
                    payload={"dt": f.dtype},
                )
        if gl is not None:
            gl = gl.astype(np.uint32)
        fl = [f.astype(np.uint32) for f in fl]

        ret["base"] = bl
        ret["gt"] = gl
        for i, f in enumerate(fn):
            ret[f"fn{i}"] = fl[i]

        return ret
