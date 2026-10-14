"""固定全帧采集；JPEG 与秒目录完成标记均原子发布。"""
import cv2
import json
import math
import os
import threading
import time
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_metrics import Metrics
from capture_buffers import ready_seconds, second_directories, buffer_usage

from .logger import logger
from .models import CaptureStatus, CaptureStatusEnum
from .source_parser import desensitize_rtsp_url


class CaptureWorker(threading.Thread):
    def __init__(self, device_no: int, source: str, source_type: str, config: Dict,
                 jpg_quality: int = 75, account: Optional[str] = None,
                 password: Optional[str] = None, max_seconds=None, max_frames=None,
                 max_disk_mb=None, stop_event=None):
        super().__init__(daemon=True)
        self.metrics = Metrics("capture")
        self.device_no, self.raw_source, self.source_type = device_no, source, source_type
        self.config, self.jpg_quality = config, jpg_quality
        self.max_seconds, self.max_frames, self.max_disk_mb = max_seconds, max_frames, max_disk_mb
        self.stop_event = stop_event or threading.Event()
        self.status = CaptureStatus(device_no=device_no, source=desensitize_rtsp_url(source),
                                    source_type=source_type, start_time=time.time())
        self.is_live_stream = source_type != "file"
        self.device_dir = Path(config["global"]["output_dir"]) / f"device_{device_no}"
        self.device_dir.mkdir(parents=True, exist_ok=True)
        self.video = None
        self.fps = 0.0
        self.frame_idx = 0
        self.current_ts = None
        self.current_seq_in_sec = 0
        self.seconds_written = 0
        self.start_real_time = 0.0
        self.start_monotonic = 0.0

    def _init_source(self, reconnect=False) -> bool:
        try:
            if self.video is not None:
                self.video.release()
            if self.source_type == "file" and not self.raw_source.lower().startswith(("http://", "https://")):
                if not Path(self.raw_source).is_file():
                    raise FileNotFoundError(f"本地视频文件不存在: {self.raw_source}")
            source = int(self.raw_source) if self.source_type == "device" else self.raw_source
            if self.source_type in ("rtsp", "stream"):
                self.video = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000])
            else:
                self.video = cv2.VideoCapture(source)
            if not self.video.isOpened():
                raise IOError(f"无法打开输入源: {self.status.source}")
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(self.fps) or self.fps <= 0:
                self.fps = float(self.config["global"]["default_fps"])
            self.status.fps = self.fps
            self.status.resolution = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.status.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.status.status = CaptureStatusEnum.RUNNING
            logger.info("[设备%d] 全帧采集启动 | %s | FPS %.3f", self.device_no, self.status.source, self.fps)
            return True
        except Exception as exc:
            self.status.last_error = str(exc).replace(self.raw_source, self.status.source)
            self.status.status = CaptureStatusEnum.RECONNECTING if reconnect else CaptureStatusEnum.ERROR
            logger.error("[设备%d] 初始化失败: %s", self.device_no, self.status.last_error)
            return False

    def _finish_second(self):
        if self.current_ts is None or not self.current_seq_in_sec:
            return
        directory = self.device_dir / str(self.current_ts)
        tmp = directory / "_ready.json.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"fps": self.fps, "frames": self.current_seq_in_sec}, handle)
        os.replace(tmp, directory / "_ready.json")
        (directory / "_writing").unlink(missing_ok=True)
        self.seconds_written += 1
        self.metrics.data.update(processed=self.seconds_written, frames=self.status.frames_captured)
        self.metrics.publish()
        logger.info("[设备%d] 秒目录完成 ts=%d 帧=%d", self.device_no,
                    self.current_ts, self.current_seq_in_sec)
        self.current_ts = None
        self.current_seq_in_sec = 0
        if os.environ.get("AIOT_PIPELINE_STATE"):
            limit = max(1, int(self.config["global"].get("max_buffer_seconds", 8)))
            while not self.stop_event.is_set():
                try:
                    progress = json.loads((Path(os.environ['AIOT_PIPELINE_STATE']) / 'tag.metrics.json').read_text(encoding='utf-8'))
                    consumed = int(progress.get('last_ts', self.start_real_time - 1))
                except (OSError, ValueError):
                    consumed = self.start_real_time - 1
                if self.config['global'].get('buffer_mode') == 'runtime':
                    # Occupied consumed files count too: storage cannot grow past the limit.
                    pending = len(second_directories(self.device_dir, self.start_real_time))
                else:
                    pending = len(ready_seconds(self.device_dir, consumed + 1))
                if pending < limit:
                    break
                self.stop_event.wait(0.05)

    def run(self):
        if not self._init_source():
            self.release()
            return
        self.start_real_time = int(os.environ.get("AIOT_PIPELINE_START_TS", str(int(time.time()))))
        self.start_monotonic = time.monotonic()
        reconnects = fails = 0
        rc = self.config["reconnect"]
        try:
            while not self.stop_event.is_set():
                with self.metrics.measure("decode"):
                    ret, frame = self.video.read()
                if not ret:
                    if not self.is_live_stream:
                        if self.status.frame_count > 0 and self.frame_idx < self.status.frame_count:
                            raise IOError(f"视频提前结束: {self.frame_idx}/{self.status.frame_count} 帧")
                        self.status.status = CaptureStatusEnum.FINISHED
                        break
                    fails += 1
                    self._finish_second()
                    if fails >= rc["max_frame_fails"]:
                        if reconnects >= rc["max_reconnect_times"]:
                            raise IOError("重连超过最大次数")
                        reconnects += 1
                        self.status.status = CaptureStatusEnum.RECONNECTING
                        logger.warning("[设备%d] 断流重连 %d", self.device_no, reconnects)
                        if self.stop_event.wait(rc["reconnect_interval"]):
                            break
                        self._init_source(reconnect=True)
                    elif self.stop_event.wait(rc["reconnect_interval"]):
                        break
                    continue
                fails = reconnects = 0
                self.status.status = CaptureStatusEnum.RUNNING
                if not self.is_live_stream:
                    # 本地文件默认尽快读完，避免采集端把整个流水线锁在视频时长。
                    # 实时流仍按真实帧率运行；可用 global.file_realtime=true 恢复文件实时节奏。
                    if self.config["global"].get("file_realtime", False):
                        delay = self.frame_idx / self.fps - (time.monotonic() - self.start_monotonic)
                        if delay > 0 and self.stop_event.wait(delay):
                            break
                    ts = int(self.start_real_time + self.frame_idx / self.fps)
                else:
                    ts = int(time.time())
                    if (self.device_dir / str(ts) / "_ready.json").exists():
                        if self.stop_event.wait(max(0.001, ts + 1 - time.time())):
                            break
                        ts = int(time.time())
                if ts != self.current_ts:
                    self._finish_second()
                    directory = self.device_dir / str(ts)
                    directory.mkdir(exist_ok=False)
                    (directory / "_writing").touch()
                    self.current_ts = ts
                directory = self.device_dir / str(ts)
                destination = directory / f"frame_{self.current_seq_in_sec:06d}.jpg"
                tmp = destination.with_suffix(".jpg.tmp")
                with self.metrics.measure("jpeg_encode"):
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])
                if not ok:
                    raise IOError("JPEG 编码失败")
                with self.metrics.measure("frame_write"):
                    encoded.tofile(str(tmp))
                    os.replace(tmp, destination)
                self.current_seq_in_sec += 1
                self.frame_idx += 1
                self.status.frames_captured += 1
                if self.is_live_stream and self.max_seconds and time.monotonic() - self.start_monotonic >= self.max_seconds:
                    self.status.status = CaptureStatusEnum.FINISHED
                    break
                if self.status.frames_captured % 30 == 0:
                    self._check_capacity()
            self._finish_second()
        except Exception as exc:
            self.status.last_error = str(exc).replace(self.raw_source, self.status.source)
            self.status.status = CaptureStatusEnum.ERROR
            logger.exception("[设备%d] 采集失败: %s", self.device_no, self.status.last_error)
        finally:
            self.metrics.publish(finished=True)
            self.release()
            logger.info("[设备%d] 已停止，帧=%d 秒=%d 状态=%s", self.device_no,
                        self.status.frames_captured, self.seconds_written, self.status.status.value)

    def _check_capacity(self):
        frames, size = buffer_usage(self.device_dir)
        if self.max_frames and frames > self.max_frames:
            raise IOError("未消费帧数超过 max_frames；停止采集以保留积压数据")
        if self.max_disk_mb and size > self.max_disk_mb * 1024 * 1024:
            raise IOError("未消费帧超过 max_disk_mb；停止采集以保留积压数据")

    def release(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.status.status not in (CaptureStatusEnum.ERROR, CaptureStatusEnum.FINISHED):
            self.status.status = CaptureStatusEnum.STOPPED

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=12)
