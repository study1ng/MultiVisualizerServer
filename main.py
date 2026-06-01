import asyncio

from fastapi import FastAPI, Query, Request
from typing import List
from upath import UPath
from fastapi.responses import JSONResponse, StreamingResponse
from method.ctdashboard import CTDashboardMethod
from method.dumbdashboard import DumbDashboardMethod
from method.normal import Normal
from methods_types import DashboardMethod, Method
from utils import is_nii, map_dict, ndarray2bytes, resolved_path
from errors import *
import io, json, zipfile, zlib, uuid, logging
import numpy as np
from numpy import ndarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
_storage: dict[str, bytes] = {}


def register_data(uid: uuid.UUID, data: dict[str, bytes], payload: dict) -> str:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for k, v in data.items():
            if not v:
                continue
            zip_file.writestr(f"{k}.bin", v)
        zip_file.writestr("payload.json", json.dumps(payload))

    _storage[str(uid)] = zip_buffer.getvalue()

    asyncio.create_task(_cleanup_storage(uid))


async def _cleanup_storage(uid: str, delay: int = 300):
    await asyncio.sleep(delay)
    if uid in _storage:
        del _storage[uid]


class AnyJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        def default_converter(o):
            return str(o)

        return json.dumps(
            content,
            default=default_converter,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


@app.exception_handler(MvisError)
async def mvis_error_handler(req: Request, exc: MvisError):
    content = {"ty": type(exc).__name__, "messages": list(exc.args), **exc.payload}

    logger.error(f"{req.url} {exc.__class__} {content}")

    if isinstance(exc, InternalError):
        status_code = 500
    else:
        status_code = 400

    return AnyJSONResponse(
        status_code=status_code,
        content=content,
    )


def preprocess_pth(p):
    if p is None:
        return None
    p = resolved_path(p)
    if not p.exists():
        raise FileError("file not found", payload={"filename": p})
    if p.suffixes == []:
        raise FileError(
            "unable to specify file format due to no ext", payload={"filename": p}
        )
    return p


@app.get("/{view_method}/")
async def entry(
    view_method: str,
    base: str | None = None,
    gt: str | None = None,
    fn: List[str] | None = Query(default_factory=list),
):
    # FIX: /view_method/base=&gt=...でbaseがNoneに変換されない
    """entry is for the first phase processing which process view-method to a dict and resolve and validate paths existence"""
    logger.info(
        f"GET: {view_method}, base: {str(base)}, gt: {str(gt)}, fn: {[str(f) for f in fn] if fn is not None else []}"
    )
    base = UPath(base) if base else None
    gt = UPath(gt) if gt else None
    fn = [UPath(f) for f in fn] if fn else None

    data, payload = entry_logic(view_method, base, gt, fn)

    uid = uuid.uuid4()
    register_data(uid, data, payload)

    return str(uid)


def entry_logic(
    view_method: str, base: UPath | None, gt: UPath | None, fn: List[UPath] | None
):
    ax, view_method = get_method(view_method)

    base = preprocess_pth(base)
    gt = preprocess_pth(gt)
    if fn is None:
        fn = []
    if not base and not gt and not fn:
        raise InternalError("there is no urls passed")
    fn = tuple(preprocess_pth(f) for f in fn)

    data = view(ax, view_method, base, gt, fn)

    dashboard_method = get_dashboard_method(base, gt, fn)
    payload: dict = dashboard_method.process(base, gt, fn)
    return data, payload


def view(ax: str, view_method: Method, base, gt, fn):
    def _view(v: ndarray | None):
        if v is None:
            return None
        match ax:
            case "axial":
                pass
            case "saggital":
                v = np.transpose(v, (1, 2, 0))
            case "coronal":
                v = np.transpose(v, (0, 2, 1))
            case _:
                raise InternalError("undefined ax", payload={"ax": ax})
        return zlib.compress(ndarray2bytes(v))

    data = view_method.process(base, gt, fn)
    data = map_dict(_view, data)
    return data


def get_method(view_method: str) -> tuple[str, Method]:
    out = dict()
    for vm in view_method.split(","):
        vm = vm.strip()
        if vm == "":
            continue
        pair = vm.split("=")
        if len(pair) != 2:
            raise InternalError(
                f"view method syntax error",
                payload={"view-method": out, "vm": vm},
            )
        k, v = pair
        out[k] = v
    if "ax" not in out:
        raise InternalError(
            "ax should be in view-method", payload={"view_method": view_method}
        )
    if "process" not in out:
        raise InternalError(
            "process should be in view-method", payload={"view_method": view_method}
        )
    process = out.pop("process")
    match process:
        case "normal":
            ax = out.pop("ax")
            if len(out) != 0:
                raise InternalError("remained parameters", payload={"remain": out})
            return (ax, Normal())
        case p:
            raise InternalError("unimplemented method", payload={"process": p})


def get_dashboard_method(
    base: UPath | None,
    gt: UPath | None,
    fn: List[UPath],
) -> DashboardMethod:
    bn = is_nii(base)
    gn = is_nii(gt)
    fni = all(map(is_nii, fn)) if fn else None
    nii = bn or gn or fni
    if nii:
        return CTDashboardMethod()

    return DumbDashboardMethod()


@app.get("/{uid}.zip")
def download(uid: str):
    data = _storage.get(uid)
    if not data:
        raise InternalError(
            "requested data don't exist or expired", payload={"uid": uid}
        )
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={uid}.zip"},
    )
