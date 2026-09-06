# -*- coding: utf-8 -*-
"""多路采集管理器：统一管理所有采集 worker 的生命周期。"""
import threading
from typing import Dict, Optional, List

from .capture_worker import CaptureWorker
from .source_parser import parse_source
from .logger import logger


class CaptureManager:
    """多路视频采集管理器，负责 worker 的创建、启停和状态查询。"""

    def __init__(self, config: Dict):
        self.config = config
        self._workers: Dict[int, CaptureWorker] = {}
        self._next_device_no = 0
        self._lock = threading.Lock()

    def start(self, source: str, sampling: float = 1.0, jpg_quality: int = 75,
              account: Optional[str] = None, password: Optional[str] = None,
              output_dir: Optional[str] = None, max_seconds: Optional[int] = None, max_frames: Optional[int] = None, max_disk_mb: Optional[int] = None, device_no: Optional[int] = None, stop_event = None) -> int:
        source_type, final_source = parse_source(source, account, password)
        with self._lock:
            if device_no is None:
                device_no = self._next_device_no
                self._next_device_no += 1
            elif device_no >= self._next_device_no:
                self._next_device_no = device_no + 1
        worker_config = dict(self.config)
        if output_dir:
            worker_config["global"] = {**worker_config["global"], "output_dir": output_dir}
        worker = CaptureWorker(
            device_no=device_no, source=final_source, source_type=source_type,
            config=worker_config, sampling=sampling, jpg_quality=jpg_quality,
            account=account, password=password,
            max_seconds=max_seconds,
            max_frames=max_frames, max_disk_mb=max_disk_mb,
            stop_event=stop_event,
        )
        with self._lock:
            self._workers[device_no] = worker
        worker.start()
        logger.info("[Manager] 设备 %d 已启动，源类型: %s", device_no, source_type)
        return device_no

    def stop(self, device_no: Optional[int] = None) -> List[int]:
        stopped = []
        with self._lock:
            if device_no is not None:
                worker = self._workers.get(device_no)
                if worker is None:
                    raise KeyError("设备 %d 不存在" % device_no)
                worker.stop()
                stopped.append(device_no)
            else:
                for dno, worker in list(self._workers.items()):
                    worker.stop()
                    stopped.append(dno)
        return stopped

    def status(self, device_no: Optional[int] = None) -> List[Dict]:
        with self._lock:
            if device_no is not None:
                worker = self._workers.get(device_no)
                if worker is None:
                    raise KeyError("设备 %d 不存在" % device_no)
                return [worker.status.to_dict()]
            return [w.status.to_dict() for w in self._workers.values()]

    @property
    def worker_count(self) -> int:
        return len(self._workers)
