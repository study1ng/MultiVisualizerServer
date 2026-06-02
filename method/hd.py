from errors import InternalError, ServerError
from method.normal import Normal
from methods_types import Method
from utils import load
from upath import UPath
import numpy as np
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


class HD(Method):
    def __init__(
        self,
        src: int | None = None,
    ):
        self.src = int(src) if src else None

    @staticmethod
    def _load(p):
        if p is None:
            return None

        if not isinstance(p, (UPath, str)):
            raise ServerError("ServerError: p should be Path or str")

        return load(p)

    def _process(self, base, gt, fn):
        loaded = Normal().process(base, gt, fn)
        bl = loaded.pop("base")
        gl = loaded.pop("gt")
        fnl = [loaded[f"fn{i}"] for i in range(len(fn))]
        if self.src is None and gt is None:
            raise InternalError(
                "no src and no gt so we can't judge use what as criteria"
            )
        if self.src is None:
            gl = np.zeros_like(gl)
            for i in range(len(fnl)):
                fnl[i] = self.dist_map(fnl[i], gl)
        else:
            for i in range(len(fnl)):
                if i == self.src:
                    fnl[i] = np.zeros_like(fnl[i])
                else:
                    fnl[i] = self.dist_map(fnl[i], fnl[self.src])
        ret = {}

        ret["base"] = bl
        ret["gt"] = gl
        for i in range(len(fnl)):
            ret[f"fn{i}"] = fnl[i]

        return ret

    def dist_map(self, result, reference):
        distmap = []
        resu = np.unique(result)
        refu = np.unique(reference)
        indices = set(resu).intersection(refu)
        for label in indices:
            distmap.append(self._dist_map(result == label, reference == label))
        return np.maximum.reduce(distmap)

    def _dist_map(self, result, reference):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(np.bool_))
        reference = np.atleast_1d(reference.astype(np.bool_))
        # binary structure
        footprint = generate_binary_structure(result.ndim, 1)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError(
                "The first supplied array does not contain any binary object."
            )
        if 0 == np.count_nonzero(reference):
            raise RuntimeError(
                "The second supplied array does not contain any binary object."
            )

        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(
            result, structure=footprint, iterations=1
        )
        reference_border = reference ^ binary_erosion(
            reference, structure=footprint, iterations=1
        )

        # compute average surface distance
        dtf = distance_transform_edt(~result_border, sampling=None)
        dts = distance_transform_edt(~reference_border, sampling=None)
        return np.maximum(dtf, dts)
