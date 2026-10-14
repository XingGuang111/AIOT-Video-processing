"""真实子进程的联动退出/停止冒烟，使用隔离设备与目录。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]


def exercise(mode):
    work = Path(tempfile.mkdtemp(prefix=f"pipeline_{mode}_", dir=ROOT / "logs"))
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    for key in ("capture_buffer", "buffer_101", "buffer_201", "annotated_buffer", "logs_dir"):
        cfg["global"][key] = str(work / key)
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    environment = os.environ.copy()
    environment.update({"AIOT_CONFIG_PATH": str(path), "PYTHONIOENCODING": "utf-8"})
    with (work / "console.log").open("w", encoding="utf-8") as log:
        root = subprocess.Popen([sys.executable, "-u", str(ROOT / "pipeline.py"), "--source",
                                 str(ROOT / "VideoCapture/demo_videos/video3.mp4"),
                                 "--device-no", "923002"], env=environment, cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
        print(f"{mode} root_pid={root.pid} work={work}", flush=True)
        deadline = time.monotonic() + 60
        state = None
        try:
            while time.monotonic() < deadline:
                paths = list((work / "logs_dir").glob("pipeline_*/children.json"))
                if paths:
                    state = paths[0].parent
                    break
                if root.poll() is not None:
                    raise AssertionError((work / "console.log").read_text(encoding="utf-8"))
                time.sleep(0.1)
            assert state is not None
            pids = json.loads((state / "children.json").read_text(encoding="utf-8"))
            if mode == "fault":
                psutil.Process(pids["inference"]).terminate()
            else:
                (state / "pipeline.stop").touch()
            result = root.wait(timeout=45)
            assert result == (1 if mode == "fault" else 130), result
            assert not any(psutil.pid_exists(pid) for pid in pids.values()), pids
            assert not list((work / "annotated_buffer").rglob("*.tmp.mp4"))
            assert not (work / "capture_buffer/device_923002/_pipeline.lock").exists()
            report = {"passed": mode, "exit_code": result, "orphan_processes": 0}
            (work / "verification.json").write_text(json.dumps(report), encoding="utf-8")
            return report
        finally:
            if root.poll() is None:
                if state:
                    (state / "pipeline.stop").touch()
                root.wait(timeout=45)


if __name__ == "__main__":
    print(json.dumps([exercise("fault"), exercise("stop")], indent=2))
