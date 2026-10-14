# -*- coding: utf-8 -*-
"""多路采集管理器：统一管理所有采集 worker 的生命周期。

系统调用链：
1. API 路由在 app.py 调用 manager.start(...)
2. manager.start() 解析 source，创建对应 CaptureWorker 实例
3. worker.start() 启动后台线程，worker.run() 真正读取帧并保存
4. app.py 中 stop/status 会进一步调用 manager.stop()/manager.status()
5. manager 通过维护 _workers 字典统一管理所有设备
"""
import threading
from pathlib import Path
from typing import Dict, Optional, List

from .capture_worker import CaptureWorker
from .source_parser import parse_source
from .logger import logger


class CaptureManager:
    """多路视频采集管理器，负责 worker 的创建、启停和状态查询。"""

    def __init__(self, config: Dict):
        # 该配置来自 config.yaml，包含全局输出目录、重连参数、清理策略等
        self.config = config
        # 所有采集线程按 device_no 作为索引保存在此字典中
        self._workers: Dict[int, CaptureWorker] = {}
        # 自动分配设备编号时的计数器，保证每次启动新任务时不会重复
        self._next_device_no = 0
        # 保证并发启动/停止/查询时不会出现字典访问冲突
        self._lock = threading.Lock()#

    def start(self, source: str, jpg_quality: Optional[int] = None,
              account: Optional[str] = None, password: Optional[str] = None,
              output_dir: Optional[str] = None, max_seconds: Optional[int] = None, max_frames: Optional[int] = None, max_disk_mb: Optional[int] = None, device_no: Optional[int] = None, stop_event=None) -> int:
        """启动一条采集任务，并返回对应的设备编号。

        调用链：
        app.py -> start_capture(req) -> manager.start(...)
        其中会解析源类型、创建 CaptureWorker、注册到 _workers、然后调用 worker.start()
        """
        # 1. 先判断输入源属于哪一类：设备号 / RTSP / 本地文件 / HTTP 视频流
        source_type, final_source = parse_source(source, account, password)

        # 2. 分配唯一 device_no，避免多路任务编号冲突
        with self._lock:
            if device_no is None:
                device_no = self._next_device_no
                self._next_device_no += 1
            elif device_no >= self._next_device_no:
                self._next_device_no = device_no + 1
            if device_no < 0:
                raise ValueError("device_no 必须非负")
            existing = self._workers.get(device_no)
            if existing and existing.is_alive():
                raise RuntimeError(f"设备 {device_no} 已在运行")

        # 3. 复制一份配置，并在需要时覆盖输出目录，保证每个 worker 的输出位置可定制
        worker_config = dict(self.config)
        root = Path(__file__).resolve().parents[2]
        directory = Path(output_dir or worker_config["global"]["output_dir"])
        worker_config["global"] = {**worker_config["global"], "output_dir": str(root / directory)}
        if jpg_quality is None:
            jpg_quality = int(worker_config.get("extract", {}).get("jpg_quality", 75))
        if not 1 <= jpg_quality <= 100:
            raise ValueError("jpg_quality 必须在 1-100")

        # 4. 创建一个真正的采集线程对象，里面保存了源地址、采样控制、限制条件等参数，worker.run() 会在新线程中循环读取帧并保存
        worker = CaptureWorker(
            device_no=device_no,
            source=final_source,
            source_type=source_type,
            config=worker_config,
            jpg_quality=jpg_quality,
            account=account,
            password=password,
            max_seconds=max_seconds,
            max_frames=max_frames,
            max_disk_mb=max_disk_mb,
            stop_event=stop_event,
        )

        # 5. 注册到管理器中，供后续 status()/stop() 统一查询和关闭
        with self._lock:
            self._workers[device_no] = worker

        # 6. 真正线程启动，run() 进入主采集循环，创建新线程并启动worker.run()，该线程会一直运行直到采集结束或被 stop() 停止
        worker.start()
        logger.info("[Manager] 设备 %d 已启动，源类型: %s", device_no, source_type)
        return device_no

    def stop(self, device_no: Optional[int] = None) -> List[int]:
        """停止一个设备或全部设备。

        调用链：
        app.py -> stop_capture(req) -> manager.stop(device_no)
        -> worker.stop() -> worker.release() -> run() 退出
        """
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
        """查询一个或全部设备的状态。

        调用链：
        app.py -> get_status(device_no) -> manager.status(device_no)
        -> worker.status.to_dict()
        """
        with self._lock:
            if device_no is not None:
                worker = self._workers.get(device_no)
                if worker is None:
                    raise KeyError("设备 %d 不存在" % device_no)
                return [worker.status.to_dict()]
            return [w.status.to_dict() for w in self._workers.values()]

    @property
    def worker_count(self) -> int:
        # 当前管理器持有的采集线程数量，可用于监控并发数
        return len(self._workers)
