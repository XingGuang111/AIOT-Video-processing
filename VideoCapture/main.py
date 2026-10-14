"""采集 CLI：固定全帧；文件结束后返回实际退出状态。"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def cmd_extract(source, device_no=0):
    from VideoCapture.videocapture import CaptureManager, logger
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    state = Path(os.environ["AIOT_PIPELINE_STATE"]) if os.environ.get("AIOT_PIPELINE_STATE") else None
    if state:
        run = json.loads((state / "run.json").read_text(encoding="utf-8"))
        config["global"]["output_dir"] = run["capture_buffer"]
        config["global"]["file_realtime"] = run.get("file_realtime", False)
        config["global"]["max_buffer_seconds"] = run.get("max_buffer_seconds", 8)
        config["global"]["buffer_mode"] = run.get("buffer_mode", "config")
    manager = CaptureManager(config)
    dno = manager.start(source=source, device_no=device_no)
    def stop(*_):
        manager.stop(dno)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)
    result = 0
    try:
        while True:
            status = manager.status(dno)[0]
            if status["status"] in ("finished", "stopped", "error"):
                result = int(status["status"] == "error")
                break
            if state and (state / "capture.stop").exists():
                stop()
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop()
    finally:
        manager.stop(dno)
        status = manager.status(dno)[0]
        result = max(result, int(status["status"] == "error"))
        if state:
            tmp = state / "capture.done.tmp"
            tmp.write_text(json.dumps({"frames": status["frames_captured"],
                                       "seconds": manager._workers[dno].seconds_written,
                                       "error": status["last_error"] if result else None}), encoding="utf-8")
            os.replace(tmp, state / "capture.done")
        logger.info("采集结束: %s", status)
    return result


def main():
    parser = argparse.ArgumentParser(description="视频采集（固定全帧 JPEG）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("server", help="启动 HTTP 采集服务")
    extract = sub.add_parser("extract")
    extract.add_argument("source")
    extract.add_argument("--device-no", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "server":
        from VideoCapture.app import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return 0
    return cmd_extract(args.source, args.device_no)


if __name__ == "__main__":
    raise SystemExit(main())
