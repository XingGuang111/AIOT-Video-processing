"""三独立进程 + 磁盘接力的秒级流水线启动器。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
import yaml

from InsightfaceModel.config_loader import load_config, ConfigError
from capture_buffers import ready_seconds
from buffer_cleanup import BufferCleaner

ROOT = Path(__file__).resolve().parent


def parse_device_no(value):
    try:
        number = int(str(value).removeprefix("device_"))
        if not 0 <= number <= 0xFFFFFFFF:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("设备号必须为非负整数或 device_N")


def _forward(name, pipe):
    try:
        for line in pipe:
            print(f"[{name}] {line.rstrip()}", flush=True)
    finally:
        pipe.close()


def _read_state(state, name):
    path = state / f"{name}.done"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _stop_children(children, state):
    for name, child in reversed(children):
        if child.poll() is not None:
            continue
        # Windows terminate 是强制退出；先让 CLI 主线程通过磁盘停止信号收尾。
        (state / f"{name}.stop").touch()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def run_pipeline(source, device_no=0, buffer_mode=None):
    config = load_config()
    g = config["global"]
    missing = [module for module in ("cv2", "numpy", "yaml", "torch", "ultralytics", "insightface", "onnxruntime")
               if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(f"当前 Python 缺少依赖 {missing}；请使用 .venv/Scripts/python.exe")
    if not source.lower().startswith(("rtsp://", "http://", "https://")) and not source.isdigit():
        source = str(Path(source).resolve())
        if not Path(source).is_file():
            raise FileNotFoundError(f"视频文件不存在: {source}")
    device_dir = Path(g["capture_buffer"]) / f"device_{device_no}"
    device_dir.mkdir(parents=True, exist_ok=True)
    lock_path = device_dir / "_pipeline.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"设备已有流水线或遗留锁：{lock_path}；确认旧进程已停止后移除锁")
    children, readers = [], []
    run_id = os.environ.get("AIOT_PIPELINE_RUN_ID", uuid.uuid4().hex)
    if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        raise ValueError("无效流水线运行标识")
    state = Path(g["logs_dir"]) / f"pipeline_{device_no}_{run_id}"
    result = 1
    started = time.monotonic()
    old_handlers = {}
    stop_requested = threading.Event()
    cleaner = None
    try:
        with os.fdopen(lock_fd, "w") as handle:
            handle.write(str(os.getpid()))
        state.mkdir(parents=True, exist_ok=True)  # 网站可在启动尚未完成时提前发布停止信号。
        min_ts = int(time.time())
        # Fast file processing can publish virtual seconds beyond wall-clock time.
        # Reserve a fresh range after all prior outputs to avoid reuse on the next run.
        output_dir = Path(g["annotated_buffer"]) / f"device_{device_no}"
        for directory in ((device_dir, output_dir, output_dir / "segments") if Path(source).is_file() else ()):
            for entry in directory.glob("*"):
                name = entry.name if entry.is_dir() else entry.stem
                if name.isdigit():
                    min_ts = max(min_ts, int(name) + 1)
        # 新运行只消费本次起点之后的秒，历史缓存/成片原样保留。
        for root, suffix in ((g["capture_buffer"], ""), (g["buffer_101"], ".bin"),
                             (g["buffer_201"], ".bin"), (g["annotated_buffer"], ".mp4")):
            directory = Path(root) / f"device_{device_no}"
            for entry in directory.glob("*" if not suffix else f"*{suffix}"):
                name = entry.name if not suffix else entry.stem
                if name.isdigit() and int(name) >= min_ts:
                    raise RuntimeError(f"本次时间范围与已有数据冲突（原样保留）：{entry}；请使用另一设备号")
        (state / "run.json").write_text(json.dumps({"capture_buffer": g["capture_buffer"],
                                                    "device_no": device_no, "start_ts": min_ts,
                                                    "buffer_mode": buffer_mode or "config",
                                                    "file_realtime": g.get("file_realtime", False),
                                                    "max_buffer_seconds": g.get("max_buffer_seconds", 8)}), encoding="utf-8")
        environment = os.environ.copy()
        if buffer_mode is not None:
            cleanup = buffer_mode == "runtime"
            config.setdefault("tag", {}).update({"cleanup_raw_frames": cleanup, "cleanup_protocol": cleanup})
            task_config = state / "config.yaml"
            task_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            environment["AIOT_CONFIG_PATH"] = str(task_config)
            print(f"[pipeline] buffer_mode={buffer_mode} cleanup_raw_frames={cleanup} cleanup_protocol={cleanup}", flush=True)

        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment.setdefault(variable, '1')
        environment.update({"AIOT_PIPELINE_STATE": str(state), "AIOT_PIPELINE_START_TS": str(min_ts),
                            "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        if buffer_mode == 'runtime':
            cleaner = BufferCleaner(config, device_no, state, min_ts)
            cleaner.sweep()
            cleaner.start()
        # 同一解释器拉起子进程，避免系统 Python 与项目虚拟环境混用。
        commands = [
            ("capture", ["VideoCapture/main.py", "extract", source, "--device-no", str(device_no)]),
            ("inference", ["InsightfaceModel/main.py", "--device-no", str(device_no)]),
            ("tag", ["VideoTag/main.py", "--device-no", str(device_no), "--follow"]),
        ]
        def request_stop(*_):
            stop_requested.set()
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for sig in signals:
            old_handlers[sig] = signal.signal(sig, request_stop)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        for name, arguments in commands:
            child = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                                     env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", creationflags=flags)
            children.append((name, child))
            reader = threading.Thread(target=_forward, args=(name, child.stdout), daemon=True)
            reader.start()
            readers.append(reader)
        (state / "children.json").write_text(json.dumps({name: child.pid for name, child in children}), encoding="utf-8")
        print(f"[pipeline] 启动 device_{device_no}，运行记录: {state}", flush=True)
        last_progress = 0.0
        while not stop_requested.is_set():
            if (state / "pipeline.stop").exists():
                stop_requested.set()
                continue
            for name, child in children:
                code = child.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"{name} 非零退出 ({code})")
                if code == 0 and not (state / f"{name}.done").exists():
                    raise RuntimeError(f"{name} 提前退出，未发布完成状态")
            if all(child.poll() is not None for _, child in children):
                result = 0
                break
            now = time.monotonic()
            if now - last_progress >= 1:
                pending = len(ready_seconds(device_dir, min_ts))
                print(f"[pipeline] 原帧待消费秒={pending} | "
                      + " ".join(f"{name}={'运行' if child.poll() is None else '完成'}" for name, child in children), flush=True)
                last_progress = now
            stop_requested.wait(0.2)
        if stop_requested.is_set():
            result = 130
            print("[pipeline] 收到停止信号，按 标注→推理→采集 收尾", flush=True)
    finally:
        _stop_children(children, state)
        if cleaner is not None:
            # All readers have stopped; incomplete runtime inputs are no longer consumed.
            for entry in device_dir.iterdir():
                if entry.name.isdigit():
                    cleaner.watermark = max(cleaner.watermark, int(entry.name))
            cleaner.close()
        for reader in readers:
            reader.join(timeout=2)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary = {name: _read_state(state, name) for name in ("capture", "inference", "tag")}
        if result == 0 and (summary["inference"].get("failed", 0)
                            or summary["tag"].get("failed", 0) or summary["tag"].get("skipped", 0)):
            result = 2
        summary.update({"exit_code": result, "elapsed_seconds": round(time.monotonic() - started, 2)})
        if state.exists():
            (state / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lock_path.unlink(missing_ok=True)
        print(f"[pipeline] 汇总 {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="秒级视频流水线（采集→推理→标注）")
    parser.add_argument("--source", required=True)
    parser.add_argument("--device-no", type=parse_device_no, default=0)
    parser.add_argument("--buffer-mode", choices=("runtime", "test"), default=None)
    args = parser.parse_args()
    try:
        return run_pipeline(args.source, args.device_no, args.buffer_mode)
    except (OSError, RuntimeError, ValueError, ConfigError) as exc:
        print(f"[pipeline] 失败: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
