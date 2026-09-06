# -*- coding: utf-8 -*-
"""缓冲区容量控制：按磁盘用量 / 帧数 / 时长限制自动清理旧帧。"""
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .logger import logger


class BufferCleaner(threading.Thread):
    """后台线程，定期检查并清理超出限制的帧文件。"""

    def __init__(self, config: Dict, daemon: bool = True):
        super().__init__(daemon=daemon)
        bc = config.get("buffer_clean", {})
        self._stop_event = threading.Event()
        self._interval = bc.get("clean_interval", 15)
        self._max_seconds = bc.get("max_seconds", 0)
        self._max_frames = bc.get("max_frames", 0)
        self._max_disk_mb = bc.get("max_disk_mb", 0)
        self._total_max_disk_mb = bc.get("total_max_disk_mb", 0)
        # 相对路径按项目根解析，与采集线程保持一致，避免依赖运行时 CWD
        self._output_dir = Path(__file__).resolve().parent.parent.parent / config["global"].get("output_dir", "extracted_frames")

    def run(self) -> None:
        logger.info("[BufferCleaner] 启动，间隔 %ds", self._interval)
        while not self._stop_event.wait(self._interval):
            try:
                self._clean()
            except Exception as e:
                logger.error("[BufferCleaner] 清理异常: %s", e)

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5)

    def _clean(self) -> None:
        if not self._output_dir.exists():
            return
        if self._max_seconds > 0:
            self._clean_by_seconds()
        if self._max_frames > 0:
            self._clean_by_frames()
        if self._max_disk_mb > 0:
            self._clean_by_disk()
        if self._total_max_disk_mb > 0:
            self._clean_by_total_disk()

    def _clean_by_seconds(self) -> None:
        now = int(time.time())
        cutoff = now - self._max_seconds
        removed = 0
        for device_dir in self._output_dir.iterdir():
            if not device_dir.is_dir():
                continue
            for ts_dir in device_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                try:
                    if int(ts_dir.name) < cutoff:
                        shutil.rmtree(ts_dir, ignore_errors=True)
                        removed += 1
                except ValueError:
                    continue
        if removed:
            logger.info("[BufferCleaner] 按时间清理 %d 个秒目录", removed)

    def _clean_by_frames(self) -> None:
        all_ts = self._collect_ts_dirs()
        total = sum(d[1] for d in all_ts)
        if total <= self._max_frames:
            return
        all_ts.sort(key=lambda x: x[0])
        removed = 0
        for ts_val, fc, ts_path in all_ts:
            if total <= self._max_frames:
                break
            shutil.rmtree(ts_path, ignore_errors=True)
            total -= fc
            removed += 1
        if removed:
            logger.info("[BufferCleaner] 按帧数清理 %d 个秒目录", removed)

    def _clean_by_disk(self) -> None:
        all_ts = self._collect_ts_dirs()
        total = sum(self._dir_size(d[2]) for d in all_ts)
        max_bytes = self._max_disk_mb * 1024 * 1024
        if total <= max_bytes:
            return
        all_ts.sort(key=lambda x: x[0])
        removed = 0
        for ts_val, _, ts_path in all_ts:
            if total <= max_bytes:
                break
            sz = self._dir_size(ts_path)
            shutil.rmtree(ts_path, ignore_errors=True)
            total -= sz
            removed += 1
        if removed:
            logger.info("[BufferCleaner] 按磁盘清理 %d 个秒目录", removed)

    def _clean_by_total_disk(self) -> None:
        """全局磁盘限制：所有 device_N 目录中优先删最旧"""
        all_ts = self._collect_ts_dirs()
        total = sum(self._dir_size(d[2]) for d in all_ts)
        max_bytes = self._total_max_disk_mb * 1024 * 1024
        if total <= max_bytes:
            return
        all_ts.sort(key=lambda x: x[0])
        removed = 0
        for ts_val, _, ts_path in all_ts:
            if total <= max_bytes:
                break
            sz = self._dir_size(ts_path)
            shutil.rmtree(ts_path, ignore_errors=True)
            total -= sz
            removed += 1
        if removed:
            logger.info("[BufferCleaner] 全局磁盘限制：清理 %d 个秒目录", removed)

    def _collect_ts_dirs(self) -> List[Tuple[int, int, Path]]:
        result = []
        if not self._output_dir.exists():
            return result
        for device_dir in self._output_dir.iterdir():
            if not device_dir.is_dir():
                continue
            for ts_dir in device_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                try:
                    ts = int(ts_dir.name)
                    fc = len(list(ts_dir.glob("*.jpg")))
                    result.append((ts, fc, ts_dir))
                except ValueError:
                    continue
        return result

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total
