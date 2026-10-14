"""网站任务管理；通过独立 pipeline 子进程及既有磁盘状态观察运行。"""
from collections import deque
from functools import lru_cache
import math
import cv2
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import ctypes
import shutil

from VideoCapture.videocapture.source_parser import desensitize_rtsp_url
from capture_buffers import second_directories
from runtime_paths import RESOURCE_ROOT, role_command

ROOT = RESOURCE_ROOT
VIDEO_NAME = re.compile(r"^[0-9]+(?:_[0-9]+)?(?:_(?:full|partial))?\.mp4$")
UPLOAD_NAME = re.compile(r"^[0-9a-f]{32}\.(?:mp4|avi|mov|mkv|webm|m4v|ts)$")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)) and code.value == 259)
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TaskService:
    def __init__(self, config):
        self.config = config
        self.paths = config["global"]
        self.log_root = Path(self.paths["logs_dir"])
        self.job_root = self.log_root / "web_jobs"
        self.upload_root = self.log_root / "web_uploads"
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._children = {}
        self._jobs = {}
        self._video_cache = {}
        self.closing = threading.Event()
        for file in self.job_root.glob("*.json"):
            job = read_json(file)
            if job and re.fullmatch(r"[0-9a-f]{32}", job.get("id", "")):
                self._jobs[job["id"]] = job

    def _save(self, job):
        destination = self.job_root / f"{job['id']}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)

    def state_dir(self, job):
        return self.log_root / f"pipeline_{job['device_no']}_{job['id']}"

    def resolve_source(self, source):
        source = source.strip()
        if source.startswith("upload:"):
            name = source.split(":", 1)[1]
            if not UPLOAD_NAME.fullmatch(name):
                raise ValueError("无效上传文件")
            file = self.upload_root / name
            if not file.is_file():
                raise ValueError("上传文件不存在，请重新上传")
            return str(file)
        if source.lower().startswith(("rtsp://", "http://", "https://")) or source.isdigit():
            return source
        file = Path(source)
        if not file.is_absolute():
            file = ROOT / file
        if not file.is_file():
            raise ValueError("视频文件不存在，请上传视频或检查路径")
        return str(file.resolve())

    def start(self, source, device_no, label=None, buffer_mode="test"):
        if buffer_mode not in ("runtime", "test"):
            raise ValueError("Invalid buffer mode")
        resolved = self.resolve_source(source)
        with self._lock:
            if self.closing.is_set():
                raise ValueError("网站正在关闭")
            for job in self._jobs.values():
                if job["device_no"] == device_no and self.snapshot(job)["running"]:
                    raise ValueError(f"设备 {device_no} 已有任务，请先停止")
            lock = Path(self.paths["capture_buffer"]) / f"device_{device_no}/_pipeline.lock"
            if lock.exists():
                raise ValueError(f"设备 {device_no} 正在被其他流水线使用，或留有未处理的运行锁")
            identifier = uuid.uuid4().hex
            job = {"id": identifier, "device_no": device_no, "source": desensitize_rtsp_url(label or source),
                   "created_at": time.time(), "requested_stop": False, "exit_code": None,
                   "buffer_mode": buffer_mode}
            self._save(job)
            environment = os.environ.copy()
            environment.update({"AIOT_PIPELINE_RUN_ID": identifier, "PYTHONIOENCODING": "utf-8",
                                "PYTHONUNBUFFERED": "1"})
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            log_path = self.job_root / f"{identifier}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(role_command("pipeline", "--source", resolved,
                                                          "--device-no", str(device_no),
                                                          "--buffer-mode", buffer_mode),
                                             cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=flags)
            except Exception:
                job["exit_code"] = 1
                self._save(job)
                raise
            job["pid"] = child.pid
            self._jobs[identifier] = job
            self._children[identifier] = child
            self._save(job)
            threading.Thread(target=self._watch, args=(identifier, child), daemon=True).start()
            return self.snapshot(job)

    def _watch(self, identifier, child):
        code = child.wait()
        with self._lock:
            job = self._jobs[identifier]
            job["exit_code"] = code
            job["ended_at"] = time.time()
            job["total_tracks"] = read_json(Path(self.paths["annotated_buffer"]) /
                f"device_{job['device_no']}" / "_follow_state.json", {}).get("total", 0)
            self._save(job)
            self._children.pop(identifier, None)

    def stop(self, identifier):
        with self._lock:
            job = self._jobs.get(identifier)
            if job is None:
                raise KeyError("任务不存在")
            if not self.snapshot(job)["running"]:
                return self.snapshot(job)
            job["requested_stop"] = True
            self._save(job)
            # 启动期间 state 还未建立时也能先建目录；本次独有 ID 不会碰到旧任务。
            directory = self.state_dir(job)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "pipeline.stop").touch()
            return self.snapshot(job)

    def list_jobs(self):
        with self._lock:
            return [self.snapshot(job) for job in sorted(self._jobs.values(),
                    key=lambda job: job["created_at"], reverse=True)]

    def get_job(self, identifier):
        with self._lock:
            job = self._jobs.get(identifier)
            if job is None:
                raise KeyError("任务不存在")
            return self.snapshot(job)

    def snapshot(self, job):
        state = self.state_dir(job)
        run = read_json(state / "run.json", {})
        summary = read_json(state / "summary.json")
        child = self._children.get(job["id"])
        code = child.poll() if child else job.get("exit_code")
        if code is None and summary:
            code = summary.get("exit_code")
        # 重启网站后用锁内 PID 判断接管既有流水线；不把历史记录永久显示为运行。
        running = bool(child and child.poll() is None)
        if child is None and code is None:
            lock = Path(self.paths["capture_buffer"]) / f"device_{job['device_no']}/_pipeline.lock"
            try:
                running = lock.read_text().strip() == str(job.get("pid")) and pid_alive(job.get("pid"))
            except OSError:
                running = False
        if running:
            status = "stopping" if job["requested_stop"] else ("running" if run else "starting")
        else:
            status = {0: "completed", 130: "stopped", 2: "partial"}.get(code, "failed")
        start_ts = run.get("start_ts", int(job["created_at"]))
        device = f"device_{job['device_no']}"
        capture_dir = Path(self.paths["capture_buffer"]) / device
        ready_seconds, pending_frames = 0, 0
        if running and capture_dir.exists():
            consumed_ts = read_json(state / "tag.metrics.json", {}).get("last_ts", start_ts - 1)
            for directory in second_directories(capture_dir, max(start_ts, int(consumed_ts) + 1)):
                metadata = read_json(directory / "_ready.json")
                if metadata:
                    ready_seconds += 1
                    pending_frames += metadata.get("frames", 0)
                else:
                    pending_frames += sum(1 for _ in directory.glob("*.jpg"))
        completed_stages = {name: read_json(state / f"{name}.done") for name in ("capture", "inference", "tag")}
        captured = completed_stages["capture"] or {}
        end_ts = start_ts + captured["seconds"] - 1 if captured.get("seconds") else None
        if 'file_realtime' not in run and job.get('ended_at'):
            # Older captures chose their own wall-clock start after the pipeline started.
            end_ts = int(job['ended_at'])
        results = self.results(job["device_no"], start_ts=start_ts, limit=1, end_ts=end_ts)
        total_results = self.result_count(job["device_no"], start_ts, end_ts)
        stages = {name: ("completed" if value is not None else
                        ("running" if running and run else ("waiting" if running else "stopped")))
                  for name, value in completed_stages.items()}
        total = read_json(Path(self.paths["annotated_buffer"]) / device / "_follow_state.json", {}).get("total", 0)
        if not running:
            total = job.get("total_tracks", total)
        ended = job.get("ended_at") or (job["created_at"] + summary["elapsed_seconds"] if summary and "elapsed_seconds" in summary else time.time())
        if not running:
            pending_frames = 0
            ready_seconds = 0
            total_results = completed_stages["tag"].get("success", total_results) if completed_stages["tag"] else total_results
        metrics = {name: read_json(state / f"{name}.metrics.json", {}) for name in ("capture", "inference", "tag")}
        full_path = next((file for file in self._videos(job['device_no'], start_ts, end_ts)
                          if file.stem.endswith(('_full', '_partial'))), None)
        full_video = self._result_item(full_path, job['device_no']) if full_path else None
        return {**job, "end_ts": end_ts, "metrics": metrics, "stage_details": completed_stages, "full_video": full_video, "status": status, "running": running, "exit_code": code, "start_ts": start_ts,
                "elapsed_seconds": round(ended - job["created_at"], 1),
                "stages": stages, "pending_seconds": ready_seconds, "pending_frames": pending_frames,
                "segments": total_results, "total_tracks": total, "latest": results[0] if results else None,
                "summary": summary}

    def _videos(self, device_no, start_ts=0, end_ts=None):
        directory = Path(self.paths["annotated_buffer"]) / f"device_{device_no}"
        now = time.monotonic()
        cached = self._video_cache.get(device_no)
        if cached is None or now - cached[0] > 0.25:
            files = sorted((file for file in list(directory.glob("*.mp4")) + list((directory / "segments").glob("*.mp4"))
                            if VIDEO_NAME.fullmatch(file.name)),
                           key=lambda file: int(file.stem.split("_")[0]), reverse=True)
            cached = (now, files)
            self._video_cache[device_no] = cached
        return [file for file in cached[1] if int(file.stem.split("_")[0]) >= start_ts
                and (end_ts is None or int(file.stem.split("_")[0]) <= end_ts)]

    def result_count(self, device_no, start_ts=0, end_ts=None):
        return sum(1 for file in self._videos(device_no, start_ts, end_ts) if not file.stem.endswith(("_full", "_partial")))

    @staticmethod
    @lru_cache(maxsize=2048)
    def _video_info(path, size, modified):
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or frames <= 0:
                raise IOError('Video metadata unavailable')
            return {'fps': fps, 'duration': frames / fps, 'frames': frames}
        finally:
            cap.release()

    def video_info(self, path):
        stat = path.stat()
        return self._video_info(str(path), stat.st_size, stat.st_mtime_ns)

    def playback_list(self, device_no, start_ts=0, end_ts=None):
        items, position = [], 0.0
        for path in reversed(self._videos(device_no, start_ts, end_ts)):
            if not path.stem.isdigit():
                continue
            info = self.video_info(path)
            items.append({'name': path.name, 'offset': position, **info})
            position += info['duration']
        return {'items': items, 'available': position}

    def results(self, device_no, start_ts=0, limit=60, end_ts=None):
        items = []
        for file in self._videos(device_no, start_ts, end_ts)[:limit]:
            item = self._result_item(file, device_no)
            if item:
                items.append(item)
        return items

    @staticmethod
    def _result_item(file, device_no):
        try:
            size = file.stat().st_size
        except FileNotFoundError:
            return None
        return {"kind": "partial" if file.stem.endswith("_partial") else "full" if file.stem.endswith("_full") else "segment",
                "name": file.name, "ts": int(file.stem.split("_")[0]), "size": size,
                "thumbnail_url": f"/api/results/{device_no}/{file.name}/thumbnail",
                "stream_url": f"/api/stream/{device_no}?segment={file.name}",
                "download_url": f"/api/results/{device_no}/{file.name}/download"}

    def video_path(self, device_no, name):
        if not VIDEO_NAME.fullmatch(name):
            raise KeyError("无效视频名称")
        file = Path(self.paths["annotated_buffer"]) / f"device_{device_no}" / name
        if not file.is_file():
            file = file.parent / "segments" / name
        if not file.is_file():
            raise KeyError("视频尚未生成或不存在")
        return file

    def devices(self):
        ids = {job["device_no"] for job in self._jobs.values()}
        for path in Path(self.paths["annotated_buffer"]).glob("device_*"):
            if path.is_dir() and path.name[7:].isdigit():
                ids.add(int(path.name[7:]))
        return sorted(ids)

    def logs(self, identifier, limit=100):
        self.get_job(identifier)
        try:
            with (self.job_root / f"{identifier}.log").open(encoding="utf-8", errors="replace") as file:
                return list(deque(file, maxlen=limit))
        except OSError:
            return []

    def clear_records(self):
        """Remove finished task records and their logs; preserve videos and sources."""
        with self._lock:
            finished = [job for job in self._jobs.values() if not self.snapshot(job)["running"]]
            for job in finished:
                identifier = job["id"]
                if not re.fullmatch(r"[0-9a-f]{32}", identifier):
                    continue
                directory = self.state_dir(job).resolve()
                if directory.parent != self.log_root.resolve():
                    raise ValueError("无效记录目录")
                for extension in ("json", "log"):
                    (self.job_root / f"{identifier}.{extension}").unlink(missing_ok=True)
                if directory.exists():
                    shutil.rmtree(directory)
                del self._jobs[identifier]
            return {"deleted": len(finished), "retained": len(self._jobs)}

    def playback_finished(self, device_no, identifier=None):
        if identifier:
            job = self._jobs.get(identifier)
            return not job or not self.snapshot(job)["running"]
        return not any(job["device_no"] == device_no and self.snapshot(job)["running"]
                       for job in list(self._jobs.values()))

    def process_events(self, identifier, limit=100):
        job = self.get_job(identifier)
        events = []
        for line in self.logs(identifier, 3000):
            match = re.search(r"\[(capture|inference|tag|pipeline)\].*?(?:\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\])?(.*)", line)
            if not match:
                continue
            stage = match.group(1)
            timestamp = re.search(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d),\d+", line)
            ts = re.search(r"ts=(\d+)", line)
            second = int(ts.group(1)) - job["start_ts"] + 1 if ts else None
            message = None
            if "秒目录完成" in line:
                message = "采集完成，等待推理"
            elif stage == "inference" and ts and "人体=" in line:
                message = "人体检测 → 人脸识别 → 协议输出完成 · " + line.split("人体=", 1)[1].strip()
                message = message.replace(" · ", " · 人体=", 1)
            elif "秒段完成" in line:
                message = "标注绘制 → 视频编码 → 一秒片段发布完成"
            elif "完整视频生成" in line:
                message = "完整标注视频合并完成"
            elif "加载 YOLO" in line or "加载 InsightFace" in line:
                message = line.split("[Engine]", 1)[-1].strip()
            elif "汇总" in line:
                message = f"处理结束 · 总耗时 {job['elapsed_seconds']} 秒 · 已发布 {job['segments']} 段"
            elif "cleanup complete" in line:
                message = "中间文件清理完成：对应秒的已消费缓存已删除"
            elif "buffer_cleanup removed=" in line:
                message = "动态清理历史或已消费的中间缓存 · " + line.strip()
            elif "cleanup delayed" in line or "cleanup pending" in line:
                message = "中间文件暂时占用，延迟清理 · " + line.strip()
            elif "buffer_mode=" in line:
                message = "运行模式：标注成功后自动清理中间文件" if "buffer_mode=runtime" in line else "测试模式：保留各阶段中间文件"
            elif "ERROR" in line or "WARNING" in line or "Error:" in line:
                message = line.strip()
            elif "启动 device_" in line:
                message = "任务启动，采集 / 推理 / 标注并行运行"
            if message:
                events.append({"stage": stage, "time": timestamp.group(1) if timestamp else "",
                               "second": second, "message": message,
                               "level": "error" if "ERROR" in line or "Error:" in line else "info"})
        return events[-limit:]

    def shutdown(self):
        self.closing.set()
        with self._lock:
            jobs = [job["id"] for job in self._jobs.values() if self.snapshot(job)["running"]]
        for identifier in jobs:
            self.stop(identifier)
        with self._lock:
            children = list(self._children.values())
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
