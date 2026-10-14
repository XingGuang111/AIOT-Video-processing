# -*- coding: utf-8 -*-
"""Annotate completed capture seconds with interpolated boxes; support offline batches."""
import os
import threading
import json
import math
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import cv2

from runtime_metrics import Metrics, write_json
from capture_buffers import ready_seconds
from .logger import logger
from . import protocol_decoder as proto
from .frame_annotations import FrameAnnotations


def _is_file_source(source: str) -> bool:
    """判断视频源是否为本地文件（不是 rtsp/http/nvr 等流地址）。"""
    s = str(source).lower()
    return not s.startswith(("rtsp://", "http://", "https://", "nvr://"))


class SegmentProcessor:
    """单设备视频标注处理器。

    职责：读取一台设备的原始视频 + 该设备的 101/201 协议检测结果，
    把检测/识别结果逐帧绘制到视频画面上（即"视频标注"），
    再按固定时长分段编码输出标注后的 mp4。
    """

    def __init__(self, device_no: int, source: str, config: Dict, follow: bool = False):
        self.device_no = device_no  # 设备编号（与协议数据目录 device_N 对应）
        self.source = source        # 视频源（本期仅支持本地视频文件路径）
        g = config["global"]
        tag = config["tag"]
        # ---- 输入/输出路径 ----
        self.buffer_101 = Path(g["buffer_101"])  # 101 协议（人体检测/跟踪）二进制数据目录
        self.buffer_201 = Path(g["buffer_201"])  # 201 协议（人脸识别）二进制数据目录
        self.out_dir = Path(g["annotated_buffer"]) / f"device_{device_no}"  # 标注视频输出目录
        # ---- 分段与绘制样式配置 ----
        self.segment_seconds = int(tag["segment_seconds"])               # 每段视频时长（秒）
        self.person_color = tuple(tag.get("person_color", [0, 255, 0]))  # 人体框颜色（默认绿）
        self.face_color = tuple(tag.get("face_color", [0, 0, 255]))      # 人脸框颜色（默认红）
        self.roi_color = tuple(tag.get("roi_color", [255, 255, 255]))    # ROI 框颜色（默认白）
        self.roi = tag.get("roi", []) or []                              # ROI 区域 [[x1,y1,x2,y2], ...]
        self.thickness = int(tag.get("line_thickness", 2))               # 线宽
        self.font_scale = float(tag.get("font_scale", 0.6))              # 字号
        self.fourcc = tag.get("fourcc", "avc1")                          # 编码格式（H.264，失败自动回退 mp4v）
        self.cfg_start_ts = int(tag.get("start_ts", 0))                  # 配置的标注起始 ts（0=自动取协议最小 ts）
        self._cum_count = 0  # 累计出现过的总人数（到当前时刻见过的最大顺序编号）
        self.follow = follow
        self.capture_dir = Path(g.get("capture_buffer", "capture_buffer")) / f"device_{device_no}"
        self.cleanup_raw_frames = bool(tag.get("cleanup_raw_frames", True))
        self.cleanup_protocol = bool(tag.get("cleanup_protocol", False))
        self._cleanup_pending = set()
        self.overwrite_protection = bool(tag.get("overwrite_protection", True))
        self.poll_interval = float(tag.get("follow_poll_interval", 0.5))
        self.pipeline_state = Path(os.environ["AIOT_PIPELINE_STATE"]) if os.environ.get("AIOT_PIPELINE_STATE") else None
        self.min_ts = int(os.environ.get("AIOT_PIPELINE_START_TS", "0"))
        self.segments = 0
        self.skipped = 0
        self.failed = 0
        self._last_ts = -1
        self._seen = set()
        self.metrics = Metrics("tag")
        self.segments_dir = self.out_dir / "segments"
        self._state_path = self.out_dir / "_follow_state.json"

    def run(self, stop_event: Optional[threading.Event] = None) -> int:
        """标注主流程：逐帧读视频 → _draw() 绘制标注 → 分段编码输出，返回生成的分段数量。

        stop_event: 外部停止信号（由 TagWorker 线程传入），置位后安全退出循环。
        """
        if self.follow:
            return self._follow_loop(stop_event or threading.Event())
        if not _is_file_source(self.source):
            logger.error("[设备%d] NVR/流模式暂未实现: %s", self.device_no, self.source)
            raise NotImplementedError("NVR/摄像头模式本期未实现，请使用本地视频文件")

        if not Path(self.source).exists():
            raise FileNotFoundError(f"视频文件不存在: {self.source}")

        # 加载该设备全部 101/201 协议数据，按秒级 ts 重组为 {ts: {"persons": [...], "faces": [...]}}
        annotations = proto.load_device(self.buffer_101, self.buffer_201, self.device_no)
        if not annotations:
            logger.warning("[设备%d] 未找到任何协议数据，标注将为空", self.device_no)
        # 标注起始时间戳：优先用配置值，否则取协议数据中的最小 ts（视频第 0 秒对应该 ts）
        start_ts = self.cfg_start_ts or (min(annotations) if annotations else 0)

        cap = cv2.VideoCapture(self.source)#打开视频文件，返回 VideoCapture 对象
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0              # 帧率（取不到时默认 25）
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 帧宽
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 帧高
        self.out_dir.mkdir(parents=True, exist_ok=True)      # 创建标注视频输出目录

        writer = None
        tmp_path = None
        final_path = None
        cur_seg = -1
        seg_count = 0
        frame_idx = 0
        try:
            while True:
                # 0) 收到外部停止信号则退出循环
                if stop_event is not None and stop_event.is_set():
                    break
                # 1) 逐帧读取视频
                ret, frame = cap.read()
                if not ret:
                    break
                # 2) 由帧号换算该帧所属的秒级时间戳 ts，再算出所属分段序号
                ts = start_ts + int(frame_idx / fps)
                seg = (ts - start_ts) // self.segment_seconds

                # 3) 跨入新分段时：关闭上一个分段并落盘，开启新分段文件
                if seg != cur_seg:
                    if writer is not None:
                        writer.release()
                        os.replace(tmp_path, final_path)  # 写完后 .tmp.mp4 原子改名为正式 .mp4
                        logger.info("[设备%d] 分段完成: %s", self.device_no, final_path.name)
                        seg_count += 1
                    cur_seg = seg
                    seg_start = start_ts + seg * self.segment_seconds
                    seg_end = seg_start + self.segment_seconds - 1
                    final_path = self.out_dir / f"{seg_start}_{seg_end}.mp4"
                    tmp_path = self.out_dir / f"{seg_start}_{seg_end}.tmp.mp4"
                    writer = self._open_writer(tmp_path, fps, width, height)

                # 4) ★★★ 视频标注在这里执行 ★★★
                #    取该帧所属秒 ts 的标注数据（无则为 None），绘制到帧画面上
                self._draw(frame, annotations.get(ts))
                # 5) 将标注后的帧写入当前分段文件
                writer.write(frame)
                frame_idx += 1

            # 视频读完（或被停止）后收尾：关闭并落盘最后一个分段
            if writer is not None:
                writer.release()
                os.replace(tmp_path, final_path)
                logger.info("[设备%d] 分段完成: %s", self.device_no, final_path.name)
                seg_count += 1
        except Exception:
            if writer is not None:
                writer.release()
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)  # 异常时删除未完成的分段临时文件
            raise
        finally:
            cap.release()

        logger.info("[设备%d] 标注完成，共 %d 段", self.device_no, seg_count)
        return seg_count

    def _follow_loop(self, stop_event: threading.Event) -> int:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        if self._state_path.exists() and not self.pipeline_state:
            saved = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._cum_count = int(saved["total"])
            self._last_ts = int(saved["last_ts"])
        logger.info("[设备%d] follow 启动，逐秒标注: %s", self.device_no, self.capture_dir)
        while not stop_event.is_set():
            self._retry_cleanup()
            if self.pipeline_state and (self.pipeline_state / "tag.stop").exists():
                break
            done = bool(self.pipeline_state and (self.pipeline_state / "inference.done").exists())
            sides = []
            for root in (self.buffer_101, self.buffer_201):
                sides.append({int(p.stem) for p in (root / f"device_{self.device_no}").glob("*.bin")
                              if p.stem.isdigit() and int(p.stem) >= self.min_ts})
            raw = {int(p.name) for p in ready_seconds(self.capture_dir, self.min_ts)}
            candidates = sorted((sides[0] | sides[1] | raw) - self._seen)
            paired = sides[0] & sides[1]
            progressed = False
            for ts in candidates:
                if stop_event.is_set():
                    break
                if ts not in paired:
                    # 推理按升序单线程发布。后续配对已存在表示前面的秒已失败；
                    # 否则等待，不能越过仍在推理的秒导致 Total 时间顺序倒退。
                    if not done and not any(newer > ts for newer in paired):
                        break
                    logger.warning("[设备%d] 跳过未配对秒 ts=%d（保留原帧/协议）", self.device_no, ts)
                    self.skipped += 1
                else:
                    try:
                        if ts <= self._last_ts and not (self.segments_dir / f"{ts}.mp4").exists():
                            raise ValueError("时间戳早于已发布状态；拒绝倒序标注")
                        with self.metrics.measure("tag_total"):
                            published = self._annotate_second(ts, stop_event)
                        self.metrics.data["processed"] += 1
                        self.metrics.publish()
                        if published:
                            self.segments += 1
                        else:
                            self.skipped += 1
                    except InterruptedError:
                        return self.segments
                    except Exception:
                        self.failed += 1
                        logger.exception("[设备%d] 标注秒失败 ts=%d（保留输入）", self.device_no, ts)
                self._seen.add(ts)
                self.metrics.data['last_ts'] = ts
                self.metrics.publish()
                progressed = True
            if done and not ((sides[0] | sides[1] | raw) - self._seen):
                break
            if not progressed:
                stop_event.wait(self.poll_interval)
        for _ in range(3):
            self._retry_cleanup()
            if not self._cleanup_pending:
                break
            time.sleep(0.1)
        if self._cleanup_pending:
            logger.warning("[device%d] cleanup pending seconds=%s", self.device_no, sorted(self._cleanup_pending))
        stopped = stop_event.is_set() or bool(self.pipeline_state and (self.pipeline_state / "tag.stop").exists())
        if not stopped:
            with self.metrics.measure("merge"):
                self._merge_video()
        self.metrics.publish(finished=True)
        if self.pipeline_state and not stopped:
            tmp = self.pipeline_state / "tag.done.tmp"
            tmp.write_text(json.dumps({"success": self.segments, "skipped": self.skipped,
                                       "failed": self.failed}), encoding="utf-8")
            os.replace(tmp, self.pipeline_state / "tag.done")
        logger.info("[设备%d] follow 结束 成功=%d 跳过=%d 失败=%d", self.device_no,
                    self.segments, self.skipped, self.failed)
        return self.segments

    def _load_second(self, ts):
        annotation = {}
        for root, number, key in ((self.buffer_101, 101, "persons"), (self.buffer_201, 201, "faces")):
            path = root / f"device_{self.device_no}" / f"{ts}.bin"
            decoded = proto.decode(path.read_bytes())
            if decoded["protocol_no"] != number or decoded["device_no"] != self.device_no or decoded["ts"] != ts:
                raise proto.ProtocolError(f"协议头与路径不匹配: {path}")
            annotation[key] = decoded["records"]
        return annotation

    def _save_follow_state(self, ts):
        write_json(self._state_path, {"last_ts": max(self._last_ts, ts), "total": self._cum_count})
        self._last_ts = max(self._last_ts, ts)

    def _cleanup_second(self, ts):
        if not (self.cleanup_raw_frames or self.cleanup_protocol):
            return
        targets = []
        if self.cleanup_raw_frames:
            targets.append(self.capture_dir / str(ts))
        if self.cleanup_protocol:
            targets.extend(root / f"device_{self.device_no}" / f"{ts}.bin"
                           for root in (self.buffer_101, self.buffer_201))
            targets.append(self.buffer_101 / f"device_{self.device_no}" / f"{ts}.frames.json")
        pending = False
        for target in targets:
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pending = True
        if pending:
            if ts not in self._cleanup_pending:
                logger.warning("[device%d] cleanup delayed ts=%d: files still in use", self.device_no, ts)
            self._cleanup_pending.add(ts)
        else:
            self._cleanup_pending.discard(ts)
            logger.info("[device%d] cleanup complete ts=%d raw=%s protocol=%s",
                        self.device_no, ts, self.cleanup_raw_frames, self.cleanup_protocol)

    def _retry_cleanup(self):
        for ts in tuple(self._cleanup_pending):
            self._cleanup_second(ts)

    @staticmethod
    def _read_frame(path):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (OSError, cv2.error):
            return None

    def _annotate_second(self, ts, stop_event):
        annotation = self._load_second(ts)
        positions_path = self.buffer_101 / f"device_{self.device_no}" / f"{ts}.frames.json"
        positions = FrameAnnotations(json.loads(positions_path.read_text(encoding='utf-8'))['frames']
                                     if positions_path.exists() else [])
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        destination = self.segments_dir / f"{ts}.mp4"
        tmp = self.segments_dir / f"{ts}.tmp.mp4"
        if destination.exists() and self.overwrite_protection:
            cap = cv2.VideoCapture(str(destination))
            try:
                if not cap.isOpened() or not cap.read()[0]:
                    raise IOError("已存在秒段不可播放；保留输入，拒绝清理")
            finally:
                cap.release()
            # 恢复上次在视频发布后、清理输入前被打断的事务。
            for person in annotation["persons"]:
                tid = int(person["track_id"])
                if 0 < tid < proto.NO_PERSON_ID:
                    self._cum_count = max(self._cum_count, tid)
            self._save_follow_state(ts)
            self._cleanup_second(ts)
            return False
        directory = self.capture_dir / str(ts)
        if self.pipeline_state and not (directory / "_ready.json").is_file():
            raise IOError("采集秒目录尚未完成")
        frames = sorted(directory.glob("frame_*.jpg"))
        if not frames:
            raise IOError("无原始 JPEG 帧")
        metadata = json.loads((directory / "_ready.json").read_text(encoding="utf-8")) if (directory / "_ready.json").exists() else {}
        if "frames" in metadata and len(frames) != metadata["frames"]:
            raise IOError("秒目录帧数量与完成标记不一致")
        # 保留原生 FPS，尾秒/非整数 FPS 不强行拉长为一秒。
        fps = float(metadata.get("fps", len(frames)))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("无效采集 FPS")
        writer = None
        written = 0
        old_total = self._cum_count
        try:
            for frame_path in frames:
                if stop_event.is_set() or (self.pipeline_state and (self.pipeline_state / "tag.stop").exists()):
                    raise InterruptedError("停止标注")
                with self.metrics.measure("image_read"):
                    frame = self._read_frame(frame_path)
                if frame is None:
                    logger.warning("[设备%d] 跳过损坏 JPEG %s", self.device_no, frame_path)
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = self._open_writer(tmp, fps, width, height)
                elif frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                with self.metrics.measure("drawing"):
                    index = int(frame_path.stem.rsplit('_', 1)[-1])
                    self._draw(frame, positions.at(index, annotation))
                with self.metrics.measure("encoding"):
                    writer.write(frame)
                self.metrics.data["frames"] += 1
                written += 1
            if writer is None or not written:
                raise IOError("该秒无可解码帧")
            writer.release()
            writer = None
            # VideoWriter.write 无返回状态；至少验证成品能解码、帧数完整。
            cap = cv2.VideoCapture(str(tmp))
            try:
                if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != written or not cap.read()[0]:
                    raise IOError("生成的秒段无法解码或帧数不完整")
            finally:
                cap.release()
            os.replace(tmp, destination)
            self._save_follow_state(ts)
            logger.info("[设备%d] 秒段完成 ts=%d 帧=%d FPS=%.3f Total=%d", self.device_no,
                        ts, written, fps, self._cum_count)
            self._cleanup_second(ts)
            return True
        except Exception:
            if not destination.exists():
                self._cum_count = old_total
            raise
        finally:
            if writer is not None:
                writer.release()
            tmp.unlink(missing_ok=True)

    def _merge_video(self, end_ts=None):
        paths = sorted((p for p in self.segments_dir.glob("*.mp4")
                        if p.stem.isdigit() and int(p.stem) >= self.min_ts
                        and (end_ts is None or int(p.stem) <= end_ts)), key=lambda p: int(p.stem))
        if not paths:
            return
        # A file with missing seconds is explicitly labelled partial.
        start, end = int(paths[0].stem), int(paths[-1].stem)
        incomplete = self.failed or self.skipped or len(paths) != end - start + 1
        name = f"{start}_{end}" + ("_partial" if incomplete else "_full") + ".mp4"
        destination = self.out_dir / name
        from video_output import concatenate_segments
        count = concatenate_segments(paths, destination,
                                     cancelled=lambda: bool(self.pipeline_state and
                                                            (self.pipeline_state / "tag.stop").exists()))
        self.metrics.data.update(full_video=name, full_frames=count, partial=bool(incomplete))
        logger.info("[设备%d] 完整视频生成: %s 帧=%d", self.device_no, name, count)

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
        """【视频标注核心方法】在一帧画面上绘制全部标注元素（原地修改 frame）。

        ann: 该帧所属秒 ts 的标注数据 {"persons": [...], "faces": [...]}，
             None 表示本秒无检测结果（此时只画 ROI 和右上角累计人数）。
        """
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
        # 人脸框（红）+ 身份 F:n；旧批处理协议中的秒内序号也兼容显示。
        for face_index, f in enumerate(sorted(ann.get("faces", []), key=lambda face: face["xyxy"][0])):
            x1, y1, x2, y2 = f["xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.face_color, self.thickness)
            cv2.putText(frame, f"F:{face_index}", (x1, max(0, y1 - 5)),
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
