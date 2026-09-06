# -*- coding: utf-8 -*-
"""服务层：推理任务设备管理与线程调度。"""
import threading
import time
from typing import Dict, List, Optional

from .logger import logger
from .model_processor import ModelProcessor


def parse_device_no(device: str) -> int:
    """'device_0' -> 0；也接受纯数字字符串/整数。"""
    s = str(device)
    if s.startswith("device_"):
        s = s[len("device_"):]
    return int(s)


class InferenceWorker(threading.Thread):
    """单设备推理线程。"""

    def __init__(self, device_no: int, config: Dict, fps: Optional[int] = None):
        super().__init__(daemon=True)
        self.device_no = device_no
        self.stop_event = threading.Event()
        self.processor = ModelProcessor(device_no, config, fps=fps)
        self.start_time = time.time()
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            self.processor.run(self.stop_event)
        except Exception as e:
            self.error = str(e)
            logger.exception("[设备%d] 推理线程异常", self.device_no)

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=5)

    def status(self) -> Dict:
        return {
            "device_no": self.device_no,
            "alive": self.is_alive(),
            "uptime": round(time.time() - self.start_time, 1),
            "error": self.error,
        }


class InferenceManager:
    """多设备推理管理器，线程安全。"""

    def __init__(self, config: Dict):
        self.config = config
        self._workers: Dict[int, InferenceWorker] = {}
        self._lock = threading.Lock()

    def start(self, device_no, fps: Optional[int] = None) -> int:
        dno = parse_device_no(device_no)
        with self._lock:
            existing = self._workers.get(dno)
            if existing and existing.is_alive():
                raise RuntimeError(f"设备 {dno} 推理任务已在运行")
            worker = InferenceWorker(dno, self.config, fps=fps)
            self._workers[dno] = worker
        worker.start()
        logger.info("[Manager] 设备 %d 推理任务已启动 (fps=%s)", dno, fps)
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

    def join_device(self, device_no, poll: float = 5.0) -> None:
        """阻塞等待某设备线程结束（供 CLI 使用）。"""
        dno = parse_device_no(device_no)
        worker = self._workers.get(dno)
        while worker and worker.is_alive():
            time.sleep(poll)
