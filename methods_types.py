from abc import ABC, abstractmethod
from upath import UPath
from numpy import ndarray
import uuid
from errors import *


class Method(ABC):
    def __hash__(self):
        if not hasattr("_hash", self):
            self._hash = hash(uuid.uuid4())
        return self._hash

    @staticmethod
    def _check(p: UPath):
        if not p.exists():
            raise FileError("unexisting path", payload={"path": p})

    @abstractmethod
    def _process(
        self, base: UPath | None, gt: UPath | None, fn: list[UPath] | None
    ) -> dict[str, ndarray]: ...

    def process(
        self, base: UPath | None, gt: UPath | None, fn: list[UPath] | None
    ) -> dict[str, ndarray]:
        """process
        return: dict[str, ndarray], its key is filename and value is transformed result
        """
        if not base and not gt and not fn:
            raise InternalError("there is nothing passed")
        if base is not None:
            self._check(base)
        if gt is not None:
            self._check(gt)
        if fn is not None:
            for f in fn:
                self._check(f)
        if None in fn:
            raise InternalError("None in fn", payload={"fn": fn})
        ret = self._process(base, gt, fn)
        if base is None:
            ret["base"] = None
        if gt is None:
            ret["gt"] = None
        return ret


class DashboardMethod(ABC):
    CONTINUOUS_GRAPH = "continuous-graph"
    SCATTER_GRAPH = "scatter-graph"
    NUMBER = "number"
    TYPE = "type"
    RELATED = "related"
    VALUE = "value"

    def __hash__(self):
        if not hasattr("_hash", self):
            self._hash = hash(uuid.uuid4())
        return self._hash

    @abstractmethod
    def process(
        self, base: UPath | None, gt: UPath | None, fn: list[UPath] | None
    ) -> dict[str, object]: ...
