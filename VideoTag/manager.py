# -*- coding: utf-8 -*-
"""服务层：标注任务调度与处理器管理。"""
import threading
import time
from typing import Dict, List, Optional

from .logger import logger
from .segment_processor import SegmentProcessor


def parse_device_no(device: str) -> int:
    s = str(device)
    if s.startswith("device_"):
        s = s[len("device_"):]
    return int(s)


class TagWorker(threading.Thread):
    """单设备标注线程。"""

    def __init__(self, device_no: int, source: str, config: Dict):
        super().__init__(daemon=True)
        self.device_no = device_no
        self.source = source
        self.stop_event = threading.Event()
        self.processor = SegmentProcessor(device_no, source, config)
        self.start_time = time.time()
        self.error: Optional[str] = None
        self.segments = 0

    def run(self) -> None:
        try:
            self.segments = self.processor.run(self.stop_event)
        except Exception as e:
            self.error = str(e)
            logger.exception("[设备%d] 标注线程异常", self.device_no)

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=5)

    def status(self) -> Dict:
        return {
            "device_no": self.device_no,
            "alive": self.is_alive(),
            "segments": self.segments,
            "uptime": round(time.time() - self.start_time, 1),
            "error": self.error,
        }


class TagManager:
    """多设备标注管理器，线程安全。"""

    def __init__(self, config: Dict):
        self.config = config
        self._workers: Dict[int, TagWorker] = {}
        self._lock = threading.Lock()

    def start(self, source: str, device_no, ) -> int:
        dno = parse_device_no(device_no)
        with self._lock:
            existing = self._workers.get(dno)
            if existing and existing.is_alive():
                raise RuntimeError(f"设备 {dno} 标注任务已在运行")
            worker = TagWorker(dno, source, self.config)
            self._workers[dno] = worker
        worker.start()
        logger.info("[Manager] 设备 %d 标注任务已启动，源: %s", dno, source)
        return dno

    def stop(self, device_no=None) -> List[int]:
        stopped = []
        with self._lock:
            if device_no is not None:
                dno = parse_device_no(device_no)
                worker = self._workers.get(dno)
                if worker is None:
                    raise KeyError(f"设备 {dno} 不存在")
                worker.stop()
                stopped.append(dno)
                del self._workers[dno]
            else:
                for dno, worker in list(self._workers.items()):
                    worker.stop()
                    stopped.append(dno)
                self._workers.clear()
        return stopped

    def status(self, device_no=None) -> List[Dict]:
        with self._lock:
            if device_no is not None:
                dno = parse_device_no(device_no)
                worker = self._workers.get(dno)
                if worker is None:
                    raise KeyError(f"设备 {dno} 不存在")
                return [worker.status()]
            return [w.status() for w in self._workers.values()]
