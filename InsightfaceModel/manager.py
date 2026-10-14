# -*- coding: utf-8 -*-
"""服务层：推理任务设备管理与线程调度。

本模块是入口层（main.py / api.py）与推理引擎（model_processor.py）之间的中间层，
结构与 VideoTag/manager.py、VideoCapture/capture_manager.py 完全同构：

    InferenceManager（管理器，线程安全）
        └─ InferenceWorker（每设备一个守护线程）
                └─ ModelProcessor（真正的推理逻辑，本模块不关心细节）

职责边界：manager 只负责"设备号 → 线程"的生命周期管理（启动、停止、查询、
防重复启动），不碰任何模型、帧、协议——那些都在 ModelProcessor 里。
"""
import threading
import time
from typing import Dict, List, Optional

from .logger import logger
from .model_processor import ModelProcessor


def parse_device_no(device: str) -> int:
    """'device_0' -> 0；也接受纯数字字符串/整数。

    API 层传入的 device_no 格式多样（"device_0" / "0" / 0），
    统一在此归一化为 int，作为 _workers 字典的键。
    """
    s = str(device)
    if s.startswith("device_"):
        s = s[len("device_"):]
    return int(s)


class InferenceWorker(threading.Thread):
    """单设备推理线程（守护线程：主进程退出时自动随之中止）。

    本身只是一个"线程壳"：run() 里把全部工作委托给 ModelProcessor，
    自己只额外维护 停止信号 / 启动时间 / 异常信息 三样状态。
    """

    def __init__(self, device_no: int, config: Dict, fps: Optional[int] = None):
        super().__init__(daemon=True)
        self.device_no = device_no
        # 停止信号：stop() 置位后，ModelProcessor 的主循环在本轮末尾退出
        self.stop_event = threading.Event()
        # 推理核心处理器（构造很轻量，模型是首次有帧时才懒加载的）
        self.processor = ModelProcessor(device_no, config, fps=fps)
        self.start_time = time.time()   # 用于 status() 计算运行时长
        self.error: Optional[str] = None  # 线程内未捕获异常的记录，供 status 查询

    def run(self) -> None:
        """线程入口（threading.Thread 约定，由 start() 自动在新线程中调用）。

        用 try 兜住所有异常：线程内异常如果不捕获会静默死掉、难以排查，
        这里记录到 self.error 并打日志，status 接口就能看到失败原因。
        """
        try:
            self.processor.run(self.stop_event)
        except Exception as e:
            self.error = str(e)
            logger.exception("[设备%d] 推理线程异常", self.device_no)

    def stop(self) -> None:
        """请求停止（线程安全）：置位停止信号并等待线程退出，最多等 5 秒。"""
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=5)

    def status(self) -> Dict:
        """返回该设备的运行状态快照（dict，可直接 JSON 序列化）。"""
        return {
            "device_no": self.device_no,
            "alive": self.is_alive(),  # 线程是否还在跑（False = 空闲回收/异常/被停止）
            "uptime": round(time.time() - self.start_time, 1),
            "error": self.error,
        }


class InferenceManager:
    """多设备推理管理器，线程安全。

    start/stop/status 可能被 CLI 主线程或 Flask 的多个请求线程并发调用，
    因此对 _workers 字典的所有读写都通过同一把 _lock 保护。
    """

    def __init__(self, config: Dict):
        self.config = config
        # 设备号(int) → InferenceWorker 的映射表
        self._workers: Dict[int, InferenceWorker] = {}
        self._lock = threading.Lock()

    def start(self, device_no, fps: Optional[int] = None) -> int:
        """启动一台设备的推理任务，返回归一化后的设备号(int)。

        参数:
            device_no: 设备编号，接受 "device_0" / "0" / 0 三种写法。
            fps:       抽帧帧率；None 时由 ModelProcessor 读 config.yaml 的
                       inference.fps，与采集帧率独立。

        异常:
            RuntimeError: 该设备已有任务在运行（单设备单任务约束）。

        流程：查重（临界区）→ 创建 worker 并登记 → 锁外启动线程。
        """
        dno = parse_device_no(device_no)
        with self._lock:
            # 单设备单任务：同设备重复 start 直接报错，避免两个线程
            # 同时消费同一 capture_buffer/device_N 目录导致协议互相覆盖
            existing = self._workers.get(dno)#获得当前设备号对应的 InferenceWorker 实例
            if existing and existing.is_alive():#如果存在并且线程还在运行
                raise RuntimeError(f"设备 {dno} 推理任务已在运行")
            worker = InferenceWorker(dno, self.config, fps=fps)#创建一个新的 InferenceWorker 实例，传入设备号、配置和帧率
            self._workers[dno] = worker
        # start() 放在锁外：线程启动可能耗时，持锁会阻塞其他设备的并发操作
        worker.start()#开启新线程，启动run()
        logger.info("[Manager] 设备 %d 推理任务已启动 (fps=%s)", dno, fps)
        return dno

    def stop(self, device_no=None) -> List[int]:
        """停止推理任务。

        参数:
            device_no: 指定设备号（格式同 start）；None 表示停止全部设备。

        返回:
            List[int]: 实际被停止的设备号列表。

        异常:
            KeyError: 指定的设备不存在（路由层会转成 404）。

        注意与 VideoCapture 的差异：停止后会把 worker 从 _workers 中**移除**
        （推理任务是一次性的，跑完即结束，无需保留状态供事后查询）。
        """
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
                # 遍历副本（list(...)）停止所有 worker，避免迭代期间字典被修改
                for dno, worker in list(self._workers.items()):
                    worker.stop()
                    stopped.append(dno)
                self._workers.clear()
        return stopped

    def status(self, device_no=None) -> List[Dict]:
        """查询推理状态。

        参数:
            device_no: 指定设备号；None 表示查询所有设备。

        返回:
            List[Dict]: 状态字典列表（含 alive/uptime/error），
            单设备查询也统一返回列表，与 VideoCapture 的 status 行为一致。

        异常:
            KeyError: 指定的设备不存在（路由层会转成 404）。
        """
        with self._lock:
            if device_no is not None:
                dno = parse_device_no(device_no)
                worker = self._workers.get(dno)
                if worker is None:
                    raise KeyError(f"设备 {dno} 不存在")
                return [worker.status()]
            return [w.status() for w in self._workers.values()]

    def join_device(self, device_no, poll: float = 5.0) -> None:
        """阻塞等待某设备线程结束（供 CLI 使用）。

        调用方会卡在 while 循环里，每 poll 秒醒一次检查线程是否存活，
        直到推理线程自然退出（空闲回收 / 异常）才返回。
        """
        dno = parse_device_no(device_no)
        worker = self._workers.get(dno)
        while worker and worker.is_alive():
            time.sleep(poll)
