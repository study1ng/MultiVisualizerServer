from errors import ServerError
from methods_types import Method
from utils import load
from upath import UPath


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

        ret["base"] = self._load(base)
        ret["gt"] = self._load(gt)
        for i, f in enumerate(fn):
            ret[f"fn{i}"] = self._load(f)
        return ret
