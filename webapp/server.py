"""FastAPI 网站：任务启停、视频上传、结果列表与兼容浏览器的 MJPEG 预览。"""
import asyncio
from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import time
import uuid
import queue
from bisect import bisect_right
from typing import Optional, Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from InsightfaceModel.config_loader import load_config
from .service import ROOT, TaskService
from .preview import PreviewReader


class StartRequest(BaseModel):
    source: str = Field(min_length=1, max_length=4096)
    device_no: int = Field(default=0, ge=0, le=0xFFFFFFFF)
    label: Optional[str] = Field(default=None, max_length=256)
    buffer_mode: Literal["runtime", "test"] = "test"


def _jpeg(frame):
    height, width = frame.shape[:2]
    if width > 1280:
        frame = cv2.resize(frame, (1280, max(1, round(height * 1280 / width))))
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise IOError("预览图编码失败")
    return encoded.tobytes()


def _first_frame(path):
    cap = cv2.VideoCapture(str(path))
    try:
        success, frame = cap.read()
        if not success:
            raise IOError("该视频无法读取预览帧")
        return _jpeg(frame)
    finally:
        cap.release()


def create_app(config=None):
    service = TaskService(config or load_config())
    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(service.shutdown)
    app = FastAPI(title="AIOT 视频流水线控制台", lifespan=lifespan)
    app.state.service = service
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def local_mutations(request: Request, call_next):
        # 网站默认仅监听本机；拒绝其他网站借浏览器向本地接口发写请求。
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
                return Response("禁止跨站写请求", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.endswith('/thumbnail'):
            response.headers['Cache-Control'] = 'private, max-age=86400'
        elif request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/overview")
    def overview():
        demos = [{"label": path.name, "source": str(path.relative_to(ROOT)).replace("\\", "/")}
                 for path in (ROOT / "VideoCapture/demo_videos").glob("*.mp4")]
        return {"jobs": service.list_jobs(), "devices": service.devices(), "samples": demos,
                "inference_fps": service.config["inference"]["fps"], "server_time": time.time()}

    @app.post("/api/tasks", status_code=201)
    def start(body: StartRequest):
        try:
            return service.start(body.source, body.device_no, body.label, body.buffer_mode)
        except ValueError as exc:
            raise HTTPException(409 if "任务" in str(exc) or "锁" in str(exc) else 400, str(exc))

    @app.get("/api/tasks/{identifier}")
    def task(identifier: str):
        try:
            return service.get_job(identifier)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @app.delete("/api/records")
    def clear_records():
        return service.clear_records()

    @app.post("/api/tasks/{identifier}/stop")
    def stop(identifier: str):
        try:
            return service.stop(identifier)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @app.get("/api/tasks/{identifier}/logs")
    def logs(identifier: str, limit: int = Query(100, ge=1, le=300)):
        try:
            return {"lines": service.logs(identifier, limit), "events": service.process_events(identifier, limit)}
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @app.post("/api/uploads", status_code=201)
    async def upload(file: UploadFile = File(...)):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}:
            raise HTTPException(400, "请上传 MP4 / AVI / MOV / MKV / WEBM 视频")
        name = uuid.uuid4().hex + extension
        path = service.upload_root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        size = 0
        try:
            with temporary.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 ** 3:
                        raise HTTPException(413, "视频超过 2 GB，请改用服务器本地路径")
                    await asyncio.to_thread(output.write, chunk)
            os.replace(temporary, path)
            def validate():
                cap = cv2.VideoCapture(str(path))
                try:
                    return cap.isOpened() and cap.read()[0]
                finally:
                    cap.release()
            if not await asyncio.to_thread(validate):
                path.unlink(missing_ok=True)
                raise HTTPException(400, "文件不是可读取的视频")
            return {"source": f"upload:{name}", "label": file.filename, "size": size}
        finally:
            temporary.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/results/{device_no}")
    def results(device_no: int, start_ts: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=300),
                job_id: Optional[str] = None):
        if not 0 <= device_no <= 0xFFFFFFFF:
            raise HTTPException(400, "无效设备编号")
        end_ts = None
        if job_id:
            try:
                job = service.get_job(job_id)
            except KeyError:
                raise HTTPException(404, '任务不存在')
            if job['device_no'] != device_no:
                raise HTTPException(400, '任务设备不匹配')
            start_ts, end_ts = job['start_ts'], job.get('end_ts')
        return {"items": service.results(device_no, start_ts, limit, end_ts),
                "total": service.result_count(device_no, start_ts, end_ts)}

    def video_path(device_no, name):
        try:
            return service.video_path(device_no, name)
        except KeyError:
            raise HTTPException(404, "视频不存在")

    @app.get("/api/results/{device_no}/{name}/download")
    def download(device_no: int, name: str):
        return FileResponse(video_path(device_no, name), media_type="video/mp4", filename=name)

    @app.get("/api/results/{device_no}/{name}/thumbnail")
    async def thumbnail(device_no: int, name: str):
        path = video_path(device_no, name)
        try:
            return Response(await asyncio.to_thread(_first_frame, path), media_type="image/jpeg")
        except IOError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/playback/{device_no}")
    def playback(device_no: int, start_ts: int = Query(0, ge=0), job_id: Optional[str] = None):
        end_ts = None
        if job_id:
            try:
                selected = service.get_job(job_id)
            except KeyError:
                raise HTTPException(404, 'Task not found')
            if selected['device_no'] != device_no:
                raise HTTPException(400, 'Device mismatch')
            start_ts, end_ts = selected['start_ts'], selected.get('end_ts')
        return service.playback_list(device_no, start_ts, end_ts)

    @app.get("/api/stream/{device_no}")
    async def stream(request: Request, device_no: int, segment: Optional[str] = None,
                     start_ts: int = Query(0, ge=0), job_id: Optional[str] = None,
                     offset: float = Query(0, ge=0, allow_inf_nan=False)):
        if not 0 <= device_no <= 0xFFFFFFFF:
            raise HTTPException(400, "无效设备编号")
        fixed_path = video_path(device_no, segment) if segment else None
        if job_id and not fixed_path:
            try:
                selected = service.get_job(job_id)
            except KeyError:
                raise HTTPException(404, 'Task not found')
            if selected['device_no'] != device_no:
                raise HTTPException(400, 'Device mismatch')
            start_ts = selected['start_ts']
        seek_target = None
        if offset:
            playlist = None if fixed_path else await asyncio.to_thread(
                service.playback_list, device_no, start_ts, selected.get('end_ts') if job_id else None)
            available = (await asyncio.to_thread(service.video_info, fixed_path))['duration'] if fixed_path else (
                playlist)['available']
            if not math.isfinite(offset) or offset >= available:
                raise HTTPException(416, 'Seek position is not generated yet')
            if playlist:
                index = max(0, bisect_right([item['offset'] for item in playlist['items']], offset) - 1)
                seek_target = playlist['items'][index]
        async def frames():
            last_ts = start_ts - 1
            buffered = False
            reader = None
            remaining, position = offset, 0.0
            if seek_target:
                last_ts = int(Path(seek_target['name']).stem) - 1
                position = seek_target['offset']
                remaining = offset - position
                buffered = True  # Seeking existing output must not wait for two future seconds.
            try:
                while not service.closing.is_set() and not await request.is_disconnected():
                    pending = [fixed_path] if fixed_path else sorted(
                        (path for path in service._videos(device_no, start_ts)
                         if not path.stem.endswith(("_full", "_partial"))
                         and int(path.stem.split("_")[0]) > last_ts),
                        key=lambda path: int(path.stem.split("_")[0]))
                    finished = service.playback_finished(device_no, job_id) if not fixed_path else True
                    if job_id and not fixed_path:
                        try:
                            selected = service.get_job(job_id)
                            if not selected["running"]:
                                pending = [path for path in pending
                                           if int(path.stem.split("_")[0]) <= (selected.get("end_ts") or selected.get("ended_at", time.time()))]
                        except KeyError:
                            break
                    if not pending or (not fixed_path and not buffered and len(pending) < 2 and not finished):
                        if finished and not pending:
                            break
                        if not pending:
                            buffered = False
                        await asyncio.sleep(0.1)
                        continue
                    buffered = True
                    path = pending[0]
                    if not fixed_path:
                        last_ts = int(path.stem.split("_")[0])
                    info = await asyncio.to_thread(service.video_info, path)
                    start_frame = min(info['frames'] - 1, int(remaining * info['fps']))
                    remaining = 0
                    reader = PreviewReader(path, _jpeg, start_frame=start_frame)
                    frame_index = start_frame
                    while not service.closing.is_set() and not await request.is_disconnected():
                        started = time.monotonic()
                        try:
                            item = await asyncio.to_thread(reader.read)
                        except queue.Empty:
                            continue
                        if item is None:
                            break
                        data, fps = item
                        interval = 1 / fps if math.isfinite(fps) and fps > 0 else 1 / 25
                        frame_index += 1
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Segment-Timestamp: " + str(last_ts).encode() + b"\r\nX-Frame-Index: " + str(frame_index).encode() + b"\r\nX-Frame-Interval: " + str(interval).encode() + b"\r\nX-Playback-Time: " + str(position + (frame_index - 1) / fps).encode() + b"\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                        # Never accelerate after slow encoding/network delivery.
                        await asyncio.sleep(max(0, interval - (time.monotonic() - started)))
                    await asyncio.to_thread(reader.close)
                    reader = None
                    position += info['duration']
                    if fixed_path:
                        position = 0
                        await asyncio.sleep(0)  # 回看片段循环播放，不额外停顿。
            finally:
                if reader is not None:
                    await asyncio.to_thread(reader.close)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
