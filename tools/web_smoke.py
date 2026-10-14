"""隔离网站 + 浏览器全链路验证（需 logs/web_browser_qa 下安装 playwright）。"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = Path(tempfile.mkdtemp(prefix="website_smoke_", dir=ROOT / "logs"))
    sample = work / "sample.mp4"
    source = cv2.VideoCapture(str(ROOT / "VideoCapture/demo_videos/video3.mp4"))
    writer = cv2.VideoWriter(str(sample), cv2.VideoWriter_fourcc(*"mp4v"), 18, (960, 544))
    assert writer.isOpened() and source.isOpened()
    try:
        for _ in range(45):
            ok, frame = source.read()
            assert ok
            writer.write(frame)
    finally:
        writer.release()
        source.release()
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    for key in ("capture_buffer", "buffer_101", "buffer_201", "annotated_buffer", "logs_dir"):
        config["global"][key] = str(work / key)
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    environment = os.environ.copy()
    environment.update({"AIOT_CONFIG_PATH": str(config_path), "PYTHONIOENCODING": "utf-8"})
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with (work / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-u", str(ROOT / "web.py"), "--port", str(port)],
                                  env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=flags)
        print(f"WEB_SMOKE={work} URL={base}", flush=True)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(base + "/api/overview", timeout=2) as response:
                        assert response.status == 200
                    break
                except OSError:
                    time.sleep(0.2)
            result = subprocess.run(["node", str(ROOT / "tools/web_browser_smoke.cjs"), base,
                                     str(work), str(sample)], cwd=ROOT, timeout=360)
            assert result.returncode == 0, f"浏览器验证失败，日志 {work}"
        finally:
            if server.poll() is None:
                server.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                server.wait(timeout=60)
    print(f"验证通过，证据：{work / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
