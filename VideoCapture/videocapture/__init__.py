# -*- coding: utf-8 -*-
"""videocapture —— 多路视频帧采集核心包。"""
from .capture_manager import CaptureManager
from .capture_worker import CaptureWorker
from .buffer_cleaner import BufferCleaner
from .source_parser import parse_source, desensitize_rtsp_url
from .models import CaptureStatus, CaptureStatusEnum
from .logger import logger

import threading
from typing import Optional


def start_capture(
    source: str,
    account: Optional[str] = None,
    password: Optional[str] = None,
    device_no: Optional[int] = None,
    output_dir: str = "capture_buffer",
    jpg_quality: int = 75,
    max_seconds: Optional[int] = None,
    max_frames: Optional[int] = None,
    max_disk_mb: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> int:
    """启动一路视频采集任务，返回设备编号。"""
    from .capture_manager import CaptureManager
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mgr = CaptureManager(config)
    return mgr.start(
        source=source, account=account, password=password,
        device_no=device_no, output_dir=output_dir,
        jpg_quality=jpg_quality, max_seconds=max_seconds,
        max_frames=max_frames, max_disk_mb=max_disk_mb,
        stop_event=stop_event,
    )


__all__ = [
    "CaptureManager",
    "CaptureWorker",
    "BufferCleaner",
    "parse_source",
    "desensitize_rtsp_url",
    "CaptureStatus",
    "CaptureStatusEnum",
    "logger",
    "start_capture",
]
