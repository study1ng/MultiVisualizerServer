import asyncio

from fastapi import FastAPI, Request
from typing import List
from pathlib import Path
from fastapi.responses import JSONResponse, StreamingResponse
from methods import DashboardMethod, Method
from utils import ndarray2bytes, resolved_path, load
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


@app.get("/{view_method}/")
async def entry(
    view_method: str,
    base: Path | None = None,
    gt: Path | None = None,
    fn: List[Path] = [],
):
    """entry is for the first phase processing which process view-method to a dict and resolve and validate paths existence"""
    logger.info(
        f"GET: {view_method}, base: {str(base)}, gt: {str(gt)}, fn: {[str(f) for f in fn]}"
    )

    view_method = get_method(view_method)

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

    base = preprocess_pth(base)
    gt = preprocess_pth(gt)
    fn = tuple(preprocess_pth(f) for f in fn)

    return str(call(view_method=view_method, base=base, gt=gt, fn=fn))


def get_method(view_method: str) -> Method:
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
    # TODO: get appropriate method
    ...


def get_dashboard_method(
    base: Path | None,
    gt: Path | None,
    fn: List[Path],
) -> DashboardMethod:
    # TODO: get appropriate method
    ...


def call(
    view_method: Method,
    base: Path | None,
    gt: Path | None,
    fn: List[Path],
) -> uuid.UUID:
    """call is a function which call appropriate function based on view_method"""
    uid = uuid.uuid4()
    dashboard_method = get_dashboard_method(base, gt, fn)
    view_method.assert_(base, gt, fn)

    def _load(p):
        if p is None:
            return None
        return load(p)

    lb = _load(base)
    lg = _load(gt)
    lf = tuple(map(_load, fn))

    data = view_method.process(lb, lg, lf)
    data: bytes = ndarray2bytes(data)
    data = zlib.compress(data)

    payload: dict = dashboard_method.process(lb, lg, lf)
    register_data(uid, data, payload)

    return uid


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
