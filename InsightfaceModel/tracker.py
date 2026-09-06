# -*- coding: utf-8 -*-
"""引擎层：ByteTrack 多目标跟踪封装 + 连续人数编号。

track_id 由 ultralytics 内置 ByteTrack 在 PersonDetector.detect(persist=True)
中持续产生，但其原始 id 会跳变。本模块负责：把原始 id 映射为从 1 递增的
连续人数编号（每出现一个新轨迹 +1），记录每个 track 最近出现时间，定时清理
长期消失轨迹，并支持设备重启/空闲回收时整体复位。
"""
import time
from typing import Dict, List

from .logger import logger


class Tracker:
    def __init__(self, tracker_cfg: str = "bytetrack.yaml", lost_timeout: float = 30.0):
        self.tracker_cfg = tracker_cfg
        self.lost_timeout = lost_timeout
        self._last_seen: Dict[int, float] = {}     # ByteTrack 原始 id -> 最近出现时间
        self._seq_map: Dict[int, int] = {}         # ByteTrack 原始 id -> 顺序人数编号
        self._next_seq = 1                         # 下一个待分配的顺序编号（从 1 起）

    def update(self, detections: List[Dict], now: float = None) -> List[Dict]:
        """登记本帧轨迹并把 track_id 原地改写为连续人数编号。

        ByteTrack 原始 id 会因抽帧断裂、目标重进等跳变，不适合直接当人数。
        本方法为每个首次出现的原始 id 分配递增编号（1,2,3…），使
        "出现一个新人 +1"。编号只增不复用，累计总人数即最大已分配编号。
        """
        now = now if now is not None else time.time()
        for d in detections:
            raw = d.get("track_id", -1)
            if raw < 0:
                continue  # ByteTrack 未分配，保持 -1（下游按未跟踪处理）
            if raw not in self._seq_map:
                self._seq_map[raw] = self._next_seq
                self._next_seq += 1
            d["track_id"] = self._seq_map[raw]  # 原地改写为顺序编号
            self._last_seen[raw] = now
        self._cleanup(now)
        return detections

    def _cleanup(self, now: float) -> None:
        stale = [tid for tid, ts in self._last_seen.items() if now - ts > self.lost_timeout]
        for tid in stale:
            del self._last_seen[tid]
            self._seq_map.pop(tid, None)  # 一并清理映射，避免长期运行内存增长
        if stale:
            logger.debug("[Tracker] 清理消失轨迹: %s", stale)

    @property
    def active_count(self) -> int:
        return len(self._last_seen)

    @property
    def total_count(self) -> int:
        """累计出现过的不同人数（等于最大已分配顺序编号）。"""
        return self._next_seq - 1

    def reset(self) -> None:
        """复位跟踪记账（设备空闲回收/重启时调用）。"""
        self._last_seen.clear()
        self._seq_map.clear()
        self._next_seq = 1
