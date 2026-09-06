# -*- coding: utf-8 -*-
"""视频帧采集 API 服务入口 —— 薄路由层，业务逻辑委托给 videocapture 包。"""
import yaml
import uvicorn
from pathlib import Path
from typing import Optional, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from videocapture import CaptureManager, BufferCleaner, logger

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

manager = CaptureManager(config)
buffer_cleaner = BufferCleaner(config)
buffer_cleaner.start()

app = FastAPI(title="视频帧采集服务", description="支持本地视频文件、RTSP 流、USB 摄像头等多种视频源，按秒级时间戳目录输出 JPEG 帧。", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    """采集请求（支持单路或多路），source 传字符串为单路，传列表为多路"""
    source: Union[str, List[str]] = Field(description="视频源：字符串（单路）或字符串列表（多路）")
    account: Optional[str] = Field(None, description="RTSP 账号（纯 IP 时必填）")
    password: Optional[str] = Field(None, description="RTSP 密码（纯 IP 时必填）")
    sampling: float = Field(1.0, description="采样间隔（秒），-1 表示全量提取")
    device_no: Optional[int] = Field(None, description="设备编号，为空自动分配")
    max_frames: Optional[int] = Field(None, description="最大帧数限制")
    max_disk_mb: Optional[int] = Field(None, description="最大磁盘占用（MB）")
    jpg_quality: int = Field(75, ge=1, le=100, description="JPEG 质量（1-100）")
    max_seconds: Optional[int] = Field(None, description="最大采集时长（秒），0 表示不限制")
    output_dir: Optional[str] = Field(None, description="输出根目录，默认 extracted_frames")



class StopRequest(BaseModel):
    """停止采集请求"""
    device_no: Optional[int] = Field(None, description="设备号，为空则停止全部")

# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.post("/api/v1/capture/start", summary="启动一路或多路视频采集", tags=["采集控制"])
def start_capture(req: StartRequest):
    try:
        sources = req.source if isinstance(req.source, list) else [req.source]
        device_nos = []
        errors = []
        for src in sources:
            try:
                dn = manager.start(
                    source=src, sampling=req.sampling, jpg_quality=req.jpg_quality,
                    account=req.account, password=req.password,
                    output_dir=req.output_dir, device_no=req.device_no,
                    max_seconds=req.max_seconds, max_frames=req.max_frames,
                    max_disk_mb=req.max_disk_mb,
                )
                device_nos.append(dn)
            except Exception as e:
                errors.append({"source": src, "error": str(e)})
        return {"code": 0, "msg": "启动完成", "data": {"device_nos": device_nos, "errors": errors}}
    except Exception as e:
        logger.exception("启动采集失败")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/capture/stop", summary="停止采集", tags=["采集控制"])
def stop_capture(req: StopRequest):
    try:
        stopped = manager.stop(req.device_no)
        if req.device_no is not None:
            return {"code": 0, "msg": "设备 %d 已停止" % req.device_no}
        return {"code": 0, "msg": "已停止 %d 个设备" % len(stopped)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/v1/capture/status", summary="查询采集状态", tags=["状态查询"])
def get_status(device_no: Optional[int] = None):
    try:
        data = manager.status(device_no)
        if device_no is not None:
            return {"code": 0, "data": data[0]}
        return {"code": 0, "data": data}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.on_event("shutdown")
def shutdown():
    buffer_cleaner.stop()
    manager.stop()

if __name__ == "__main__":
    logger.info("🚀 视频帧采集服务启动，监听端口 8000")
    logger.info("📖 接口文档地址：http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
