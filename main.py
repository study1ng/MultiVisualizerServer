import asyncio

from fastapi import FastAPI, Request
from typing import List
from pathlib import Path
from fastapi.responses import JSONResponse, StreamingResponse
from method.dumbdashboard import DumbDashboardMethod
from method.normal import Normal
from methods_types import DashboardMethod, Method
from utils import ndarray2bytes, resolved_path
from errors import *
import io, json, zipfile, zlib, uuid, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
_storage: dict[str, bytes] = {}


def register_data(uid: uuid.UUID, data: bytes, payload: dict) -> str:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("data.bin", data)
        zip_file.writestr("payload.json", json.dumps(payload))

    _storage[uid] = zip_buffer.getvalue()

    asyncio.create_task(_cleanup_storage(uid))


async def _cleanup_storage(uid: str, delay: int = 300):
    await asyncio.sleep(delay)
    if uid in _storage:
        del _storage[uid]


@app.exception_handler(MvisError)
async def mvis_error_handler(req: Request, exc: MvisError):
    content = {"messages": list(exc.args), **exc.payload}

    logger.error(f"{req.url} {exc.__class__} {content}")

    if isinstance(exc, InternalError):
        status_code = 500
    else:
        status_code = 400

    return JSONResponse(
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
    base: Path | None = None,
    gt: Path | None = None,
    fn: List[Path] | None = None,
):
    """entry is for the first phase processing which process view-method to a dict and resolve and validate paths existence"""
    logger.info(
        f"GET: {view_method}, base: {str(base)}, gt: {str(gt)}, fn: {[str(f) for f in fn]}"
    )

    view_method = get_method(view_method)

    base = preprocess_pth(base)
    gt = preprocess_pth(gt)
    if fn is None:
        fn = []
    fn = tuple(preprocess_pth(f) for f in fn)

    data = view(view_method, base, gt, fn)

    dashboard_method = get_dashboard_method(base, gt, fn)
    payload: dict = dashboard_method.process(base, gt, fn)

    uid = uuid.uuid4()
    register_data(uid, data, payload)

    return str(uid)


def view(view_method: Method, base, gt, fn):
    data = view_method.process(base, gt, fn)
    data: bytes = ndarray2bytes(data)
    data = zlib.compress(data)
    return data


def get_method(view_method: str) -> tuple[str, Method]:
    out = dict()
    for vm in view_method.split(","):
        for kv in vm.split("=", maxsplit=1):
            if len(kv) != 2:
                raise InternalError(
                    f"view method syntax error",
                    payload={"view-method": out, "vm": vm},
                )
            k, v = kv
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
    base: Path | None,
    gt: Path | None,
    fn: List[Path],
) -> DashboardMethod:
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
