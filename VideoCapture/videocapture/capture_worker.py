import cv2
import math
import shutil
import time
import threading
from pathlib import Path
from typing import Dict, Optional
from .logger import logger
from .models import CaptureStatus, CaptureStatusEnum
from .source_parser import desensitize_rtsp_url

class CaptureWorker(threading.Thread):
    def __init__(self, device_no: int, source: str, source_type: str, config: Dict,
                 sampling: float = 1.0, jpg_quality: int = 75,
                 account: Optional[str] = None, password: Optional[str] = None, max_seconds: Optional[int] = None, max_frames: Optional[int] = None, max_disk_mb: Optional[int] = None, stop_event = None):
        super().__init__(daemon=True)
        self.device_no = device_no
        self.raw_source = source
        self.source_type = source_type
        self.config = config
        self.sampling = sampling
        self.jpg_quality = jpg_quality
        self.account = account
        self.password = password
        self.max_seconds = max_seconds
        self.max_frames = max_frames
        self.max_disk_mb = max_disk_mb
        self.stop_event = stop_event or threading.Event()
        self.lock = threading.Lock()

        # 状态初始化
        self.status = CaptureStatus(
            device_no=device_no,
            source=desensitize_rtsp_url(source),
            source_type=source_type,
            start_time=time.time()
        )
        self.video = None
        self.is_live_stream = source_type in ["rtsp", "device"]
        self.fps = 0.0
        self.frame_step = 1
        self.start_real_time = 0.0
        self.frame_idx = 0
        self.current_ts = 0
        self.current_seq_in_sec = 0

        # 重连配置
        self.max_reconnect = config["reconnect"]["max_reconnect_times"]
        self.reconnect_interval = config["reconnect"]["reconnect_interval"]
        self.max_frame_fails = config["reconnect"]["max_frame_fails"]
        self.fail_count = 0
        self.reconnect_count = 0

        # 输出目录：相对路径按项目根(videocapture→VideoCapture→根)解析，避免依赖运行时 CWD
        self.output_dir = Path(__file__).resolve().parent.parent.parent / config["global"]["output_dir"]
        self.device_dir = self.output_dir / f"device_{device_no}"
        self.device_dir.mkdir(parents=True, exist_ok=True)

    def _init_source(self) -> bool:
        """初始化视频源"""
        try:
            if self.video is not None:
                try:
                    self.video.release()
                except:
                    pass

            if self.source_type == "device":
                self.video = cv2.VideoCapture(int(self.raw_source))
            else:
                if self.source_type == "file" and not self.raw_source.lower().startswith(("http://", "https://")):
                    if not Path(self.raw_source).exists():
                        raise FileNotFoundError(f"本地视频文件不存在: {self.raw_source}")
                self.video = cv2.VideoCapture(self.raw_source)

            if not self.video.isOpened():
                reason = "文件不存在或格式不支持" if self.source_type == "file" else "连接失败或地址不可达"
                raise IOError(f"无法打开输入源 ({reason}): {self.raw_source}")

            # 实时流优化
            if self.is_live_stream:
                self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.video.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)

            # 获取视频参数
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = self.config["global"]["default_fps"]
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.status.resolution = (width, height)
            self.status.fps = self.fps
            self.status.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算采样步长
            if self.sampling != -1:
                self.frame_step = math.ceil(self.sampling * self.fps)
                if self.frame_step <= 0:
                    self.frame_step = 1
            else:
                self.frame_step = 1

            self.status.status = CaptureStatusEnum.RUNNING
            logger.info(f"[设备{self.device_no}] 启动成功 | 类型: {self.source_type} | 地址: {self.status.source} | 分辨率: {width}x{height} | FPS: {self.fps:.1f}")
            self.fail_count = 0
            self.reconnect_count = 0
            return True
        except Exception as e:
            self.status.last_error = str(e)
            self.status.status = CaptureStatusEnum.ERROR
            logger.error(f"[设备{self.device_no}] 初始化失败: {e}")
            return False

    def run(self):
        """主采集循环"""
        if not self._init_source():
            return

        self.start_real_time = time.time()
        self.current_ts = 0
        self.current_seq_in_sec = 0

        while not self.stop_event.is_set():
            try:
                with self.lock:
                    if self.stop_event.is_set():
                        break
                    ret, frame = self.video.read()
                    if not ret:
                        self.fail_count += 1
                        self.status.last_error = f"读帧失败第{self.fail_count}次"
                        if self.source_type == "file":
                            if self.fail_count >= 3:
                                logger.info(f"[设备{self.device_no}] 视频读取完成，共采集{self.status.frames_captured}帧")
                                self.status.status = CaptureStatusEnum.FINISHED
                                break
                            time.sleep(0.1)
                            continue
                        # 流模式重试重连
                        if self.fail_count >= self.max_frame_fails:
                            if self.reconnect_count >= self.max_reconnect:
                                logger.error(f"[设备{self.device_no}] 重连超过最大次数，停止采集")
                                self.status.status = CaptureStatusEnum.ERROR
                                break
                            self.status.status = CaptureStatusEnum.RECONNECTING
                            logger.warning(f"[设备{self.device_no}] 连续读帧失败，第{self.reconnect_count+1}次重连")
                            time.sleep(self.reconnect_interval)
                            if self._init_source():
                                continue
                            else:
                                self.reconnect_count += 1
                                continue
                        time.sleep(self.reconnect_interval)
                        continue

                # 重置失败计数
                self.fail_count = 0

                # 限速（文件模式全量提取时模拟真实播放速度）
                if self.source_type == "file" and self.sampling == -1:
                    expected_elapsed = self.frame_idx / self.fps
                    actual_elapsed = time.time() - self.start_real_time
                    sleep_time = expected_elapsed - actual_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # 计算时间戳
                if self.is_live_stream:
                    ts = int(time.time())
                else:
                    ts = int(self.start_real_time + (self.frame_idx / self.fps))

                if ts != self.current_ts:
                    self.current_ts = ts
                    self.current_seq_in_sec = 0

                # 保存帧
                ts_dir = self.device_dir / str(ts)
                ts_dir.mkdir(exist_ok=True)
                frame_path = ts_dir / f"frame_{self.current_seq_in_sec:06d}.jpg"
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
                cv2.imencode(".jpg", frame, encode_param)[1].tofile(str(frame_path))

                # 更新状态
                self.status.frames_captured += 1
                self.current_seq_in_sec += 1

                # 实时流超时检查
                if self.is_live_stream and self.max_seconds:
                    if time.time() - self.start_real_time >= self.max_seconds:
                        logger.info(f"[设备{self.device_no}] 达到最大采集时长 {self.max_seconds}s，自动停止")
                        self.status.status = CaptureStatusEnum.FINISHED
                        break

                # 帧数 / 磁盘限制（每 30 帧检查一次）
                if self.status.frames_captured % 30 == 0:
                    if self.max_frames:
                        self._enforce_max_frames()
                    if self.max_disk_mb:
                        self._enforce_max_disk()

                # 采样跳帧
                if self.sampling != -1:
                    self.frame_idx += self.frame_step
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                else:
                    self.frame_idx += 1

            except Exception as e:
                self.status.last_error = str(e)
                logger.error(f"[设备{self.device_no}] 采集异常: {e}")
                time.sleep(0.5)

        self.release()
        logger.info(f"[设备{self.device_no}] 已停止，共采集{self.status.frames_captured}帧")

    def release(self):
        """释放资源"""
        if self.video is not None:
            try:
                self.video.release()
            except:
                pass
        cv2.destroyAllWindows()
        self.status.status = CaptureStatusEnum.STOPPED

    def _enforce_max_frames(self):
        """帧数超限则删除最旧秒目录"""
        ts_dirs = sorted(
            [d for d in self.device_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        total = sum(len(list(d.glob("*.jpg"))) for d in ts_dirs)
        while total > self.max_frames and len(ts_dirs) > 1:
            oldest = ts_dirs.pop(0)
            count = len(list(oldest.glob("*.jpg")))
            shutil.rmtree(oldest, ignore_errors=True)
            total -= count
            logger.info(f"[设备{self.device_no}] 帧数限制：删除 {oldest.name}（{count} 帧）")

    def _enforce_max_disk(self):
        """磁盘超限则删除最旧秒目录"""
        ts_dirs = sorted(
            [d for d in self.device_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        max_bytes = self.max_disk_mb * 1024 * 1024
        total = sum(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in ts_dirs)
        while total > max_bytes and len(ts_dirs) > 1:
            oldest = ts_dirs.pop(0)
            sz = sum(f.stat().st_size for f in oldest.rglob("*") if f.is_file())
            shutil.rmtree(oldest, ignore_errors=True)
            total -= sz
            logger.info(f"[设备{self.device_no}] 磁盘限制：删除 {oldest.name}（{sz/1024/1024:.1f}MB）")

    def stop(self):
        """停止采集"""
        self.stop_event.set()
        with self.lock:
            self.release()
        if self.is_alive():
            self.join(timeout=3)
