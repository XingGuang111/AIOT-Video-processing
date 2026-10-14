# -*- coding: utf-8 -*-
"""视频帧采集 API 服务入口 —— 薄路由层，业务逻辑委托给 videocapture 包。

调用链概览：
1. 程序启动时，读取 config.yaml，初始化 CaptureManager 和 BufferCleaner。
2. 外部通过 HTTP 接口调用 /api/v1/capture/start、/stop、/status。
3. 路由函数内部调用 manager.start / manager.stop / manager.status。
4. manager 会创建并启动 CaptureWorker 线程，真正执行 OpenCV 读取视频帧，并按秒目录写出 JPEG。
5. BufferCleaner 后台线程持续清理超出磁盘/帧数/时长限制的旧目录。
"""
import yaml
import uvicorn
from pathlib import Path
from typing import Optional, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from videocapture import CaptureManager, BufferCleaner, logger

# ---------------------------------------------------------------------------
# 1. 配置加载：读取 YAML 配置文件，供 manager 和 buffer_cleaner 使用
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    # YAML 中保存了全局输出目录、重连策略、缓存清理参数等。
    config = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# 2. 业务对象初始化：核心管理器和后台清理线程都在这里创建
# ---------------------------------------------------------------------------
# CaptureManager 负责统一管理 N 个采集任务，每个任务对应一个 CaptureWorker 线程。
manager = CaptureManager(config)
# BufferCleaner 是一个后台线程，定期扫描 extracted_frames 下的数据目录并清理超限历史数据。
buffer_cleaner = BufferCleaner(config)
# 未消费的帧不能按年龄/磁盘阈值删除；正常消费后由 VideoTag 清理。

# ---------------------------------------------------------------------------
# 3. FastAPI 应用初始化：定义接口规范并打开跨域支持
# ---------------------------------------------------------------------------
app = FastAPI(
    title="视频帧采集服务",
    description="支持本地视频文件、RTSP 流、USB 摄像头等多种视频源，按秒级时间戳目录输出 JPEG 帧。",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 4. 请求体模型：定义 HTTP 请求参数格式
# ---------------------------------------------------------------------------
class StartRequest(BaseModel): # 接口处理函数
    """采集请求（支持单路或多路），source 传字符串为单路，传列表为多路。

    这个对象会被 FastAPI 自动解析 JSON 请求体；后续路由函数直接使用 req.xxx。
    """
    source: Union[str, List[str]] = Field(description="视频源：字符串（单路）或字符串列表（多路）")
    account: Optional[str] = Field(None, description="RTSP 账号（纯 IP 时必填）")
    password: Optional[str] = Field(None, description="RTSP 密码（纯 IP 时必填）")
    device_no: Optional[int] = Field(None, description="设备编号，为空自动分配")
    max_frames: Optional[int] = Field(None, description="最大帧数限制")
    max_disk_mb: Optional[int] = Field(None, description="最大磁盘占用（MB）")
    jpg_quality: Optional[int] = Field(None, ge=1, le=100, description="JPEG 质量（1-100），省略读配置")
    max_seconds: Optional[int] = Field(None, description="最大采集时长（秒），0 表示不限制")
    output_dir: Optional[str] = Field(None, description="输出根目录，省略读配置")

    class Config:
        extra = "forbid"  # 拒绝已删除的 sampling/mode 等参数，不静默忽略。


class StopRequest(BaseModel):
    """停止采集请求。

    传入 device_no 后，manager.stop(device_no) 会仅停止一台设备；不传则停止全部。
    """
    device_no: Optional[int] = Field(None, description="设备号，为空则停止全部")

# ---------------------------------------------------------------------------
# 5. 路由层：HTTP 接口 -> 调用 CaptureManager
# ---------------------------------------------------------------------------
@app.post("/api/v1/capture/start", summary="启动一路或多路视频采集", tags=["采集控制"])
def start_capture(req: StartRequest):#调用start_capture 是接口处理函数，解析成功后，req就是一个 StartRequest 对象，里面包含了请求体的所有参数。
    try:
        # 如果 source 是字符串，就包装成单元素列表；如果是列表，就保留原列表。
        sources = req.source if isinstance(req.source, list) else [req.source]
        device_nos = []
        errors = []

        for src in sources:
            try:
                # 调用 manager.start()，真正创建 CaptureWorker 并异步启动采集线程。
                dn = manager.start(
                    source=src,
                    jpg_quality=req.jpg_quality,
                    account=req.account,
                    password=req.password,
                    output_dir=req.output_dir,
                    device_no=req.device_no,
                    max_seconds=req.max_seconds,
                    max_frames=req.max_frames,
                    max_disk_mb=req.max_disk_mb,
                )
                device_nos.append(dn)
            except Exception as e:
                # 某一路采集失败不影响其他路采集，统一收集到 errors 列表中返回给调用方。
                errors.append({"source": src, "error": str(e)})

        return {"code": 0, "msg": "启动完成", "data": {"device_nos": device_nos, "errors": errors}}
    except Exception as e:
        logger.exception("启动采集失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/capture/stop", summary="停止采集", tags=["采集控制"])
def stop_capture(req: StopRequest):
    """停止指定设备或全部设备。"""
    try:
        # manager.stop() 会调用每个 worker.stop()，设置 stop_event 并释放视频资源。
        stopped = manager.stop(req.device_no)
        if req.device_no is not None:
            return {"code": 0, "msg": "设备 %d 已停止" % req.device_no}
        return {"code": 0, "msg": "已停止 %d 个设备" % len(stopped)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/capture/status", summary="查询采集状态", tags=["状态查询"])
def get_status(device_no: Optional[int] = None):
    """查询某设备状态或全部设备状态。"""
    try:
        # manager.status() 会返回每个 worker 的状态对象，包括帧数、分辨率、状态枚举等。
        data = manager.status(device_no)
        if device_no is not None:
            return {"code": 0, "data": data[0]}
        return {"code": 0, "data": data}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.on_event("shutdown")
def shutdown():
    """服务停止时，清理后台线程并关闭所有采集任务。"""
    # 先停止缓存清理线程，再关闭采集器中的所有 worker。
    buffer_cleaner.stop()
    manager.stop()


if __name__ == "__main__":
    # 直接运行这个脚本时，启动 Uvicorn 服务。
    logger.info("🚀 视频帧采集服务启动，监听端口 8000")
    logger.info("📖 接口文档地址：http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
