# -*- coding: utf-8 -*-
"""引擎层：视频分段读取、帧标注绘制、H.264 编码输出。"""
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2

from .logger import logger
from . import protocol_decoder as proto


def _is_file_source(source: str) -> bool:
    s = str(source).lower()
    return not s.startswith(("rtsp://", "http://", "https://", "nvr://"))


class SegmentProcessor:
    """单设备视频标注处理器。"""

    def __init__(self, device_no: int, source: str, config: Dict):
        self.device_no = device_no
        self.source = source
        g = config["global"]
        tag = config["tag"]
        self.buffer_101 = Path(g["buffer_101"])
        self.buffer_201 = Path(g["buffer_201"])
        self.out_dir = Path(g["annotated_buffer"]) / f"device_{device_no}"
        self.segment_seconds = int(tag["segment_seconds"])
        self.person_color = tuple(tag.get("person_color", [0, 255, 0]))
        self.face_color = tuple(tag.get("face_color", [0, 0, 255]))
        self.roi_color = tuple(tag.get("roi_color", [255, 255, 255]))
        self.roi = tag.get("roi", []) or []
        self.thickness = int(tag.get("line_thickness", 2))
        self.font_scale = float(tag.get("font_scale", 0.6))
        self.fourcc = tag.get("fourcc", "avc1")
        self.cfg_start_ts = int(tag.get("start_ts", 0))
        self._cum_count = 0  # 累计出现过的总人数（到当前时刻见过的最大顺序编号）

    def run(self, stop_event: Optional[threading.Event] = None) -> int:
        """执行标注，返回生成的分段数量。"""
        if not _is_file_source(self.source):
            logger.error("[设备%d] NVR/流模式暂未实现: %s", self.device_no, self.source)
            raise NotImplementedError("NVR/摄像头模式本期未实现，请使用本地视频文件")

        if not Path(self.source).exists():
            raise FileNotFoundError(f"视频文件不存在: {self.source}")

        annotations = proto.load_device(self.buffer_101, self.buffer_201, self.device_no)
        if not annotations:
            logger.warning("[设备%d] 未找到任何协议数据，标注将为空", self.device_no)
        start_ts = self.cfg_start_ts or (min(annotations) if annotations else 0)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        writer = None
        tmp_path = None
        final_path = None
        cur_seg = -1
        seg_count = 0
        frame_idx = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                ts = start_ts + int(frame_idx / fps)
                seg = (ts - start_ts) // self.segment_seconds

                if seg != cur_seg:
                    if writer is not None:
                        writer.release()
                        os.replace(tmp_path, final_path)
                        logger.info("[设备%d] 分段完成: %s", self.device_no, final_path.name)
                        seg_count += 1
                    cur_seg = seg
                    seg_start = start_ts + seg * self.segment_seconds
                    seg_end = seg_start + self.segment_seconds - 1
                    final_path = self.out_dir / f"{seg_start}_{seg_end}.mp4"
                    tmp_path = self.out_dir / f"{seg_start}_{seg_end}.tmp.mp4"
                    writer = self._open_writer(tmp_path, fps, width, height)

                self._draw(frame, annotations.get(ts))
                writer.write(frame)
                frame_idx += 1

            if writer is not None:
                writer.release()
                os.replace(tmp_path, final_path)
                logger.info("[设备%d] 分段完成: %s", self.device_no, final_path.name)
                seg_count += 1
        except Exception:
            if writer is not None:
                writer.release()
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
            raise
        finally:
            cap.release()

        logger.info("[设备%d] 标注完成，共 %d 段", self.device_no, seg_count)
        return seg_count

    def _open_writer(self, tmp_path, fps, width, height):
        """按配置 fourcc 打开 VideoWriter，失败则自动回退 mp4v。"""
        candidates = [self.fourcc]
        if self.fourcc != "mp4v":
            candidates.append("mp4v")
        for fcc in candidates:
            writer = cv2.VideoWriter(
                str(tmp_path), cv2.VideoWriter_fourcc(*fcc), fps, (width, height))
            if writer.isOpened():
                if fcc != self.fourcc:
                    logger.warning("[设备%d] 编码器 %s 不可用，已回退 %s",
                                   self.device_no, self.fourcc, fcc)
                return writer
            writer.release()
        raise IOError(f"VideoWriter 初始化失败，尝试过: {candidates}")

    def _draw(self, frame, ann: Optional[Dict]) -> None:
        # ROI（白）
        for r in self.roi:
            x1, y1, x2, y2 = (int(v) for v in r)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.roi_color, self.thickness)
        # 累计人数：随视频时间推进取"见过的最大顺序编号"（排除未跟踪占位 id）
        if ann:
            for p in ann.get("persons", []):
                tid = int(p["track_id"])
                if 0 < tid < proto.NO_PERSON_ID:
                    self._cum_count = max(self._cum_count, tid)
        self._draw_count(frame)  # 无论本秒有无检测，右上角始终显示当前累计值
        if not ann:
            return
        # 人体框（绿）+ 目标 ID
        for p in ann.get("persons", []):
            x1, y1, x2, y2 = p["xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.person_color, self.thickness)
            cv2.putText(frame, f"ID:{p['track_id']}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.person_color, self.thickness)
        # 人脸框（红）+ 人脸 ID（face_id == person_id，跟随人稳定；未关联人体时只画框不画 label）
        for f in ann.get("faces", []):
            x1, y1, x2, y2 = f["xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.face_color, self.thickness)
            fid = int(f.get("person_id", f.get("face_id", proto.NO_PERSON_ID)))
            if fid != proto.NO_PERSON_ID:
                cv2.putText(frame, f"F:{fid}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.face_color, self.thickness)

    def _draw_count(self, frame) -> None:
        """在画面右上角绘制累计总人数，黑底描边保证任意背景下可读。"""
        text = f"Total:{self._cum_count}"
        scale = self.font_scale * 1.5
        thick = max(1, self.thickness)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        h, w = frame.shape[:2]
        x = max(0, w - tw - 10)
        y = th + 10
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
