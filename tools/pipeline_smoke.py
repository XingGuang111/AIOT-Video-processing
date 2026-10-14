"""隔离目录中的真实模型端到端冒烟；不消费项目已有缓存。

用法: .venv/Scripts/python.exe tools/pipeline_smoke.py --seconds 30
验收帧守恒、秒段解码、原生 FPS、原帧清理、提前接力与正常排空。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.2)
    parser.add_argument("--source", default=str(ROOT / "VideoCapture/demo_videos/video1.mp4"))
    parser.add_argument("--buffer-mode", choices=("runtime", "test"), default=None)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="pipeline_smoke_", dir=ROOT / "logs"))
    source = work / "sample.mp4"
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise IOError(f"无法打开样例视频 {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise IOError("无法创建冒烟视频")
    count = 0
    try:
        for _ in range(round(args.seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
    finally:
        cap.release()
        writer.release()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    for name in ("capture_buffer", "buffer_101", "buffer_201", "annotated_buffer", "logs_dir"):
        cfg["global"][name] = str(work / name)
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    environment = os.environ.copy()
    environment.update({"AIOT_CONFIG_PATH": str(config_path), "PYTHONIOENCODING": "utf-8"})
    log_path = work / "console.log"
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "pipeline.py"),
                                  "--source", str(source), "--device-no", "923001",
                                  *(["--buffer-mode", args.buffer_mode] if args.buffer_mode else [])],
                                 cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print(f"SMOKE_WORK={work}", flush=True)
        print(f"PIPELINE_PID={child.pid} input_frames={count} fps={fps}", flush=True)
        observed_dynamic_cleanup = False
        max_pending = 0
        try:
            deadline = time.monotonic() + 900
            while child.poll() is None:
                assert time.monotonic() < deadline, 'pipeline timeout'
                raw_root = work / 'capture_buffer/device_923001'
                max_pending = max(max_pending, len(list(raw_root.glob('[0-9]*'))))
                if args.buffer_mode == 'runtime':
                    for segment in (work / 'annotated_buffer/device_923001/segments').glob('[0-9]*.mp4'):
                        ts = segment.stem
                        if not (raw_root / ts).exists() and not (work / f'buffer_101/device_923001/{ts}.bin').exists():
                            observed_dynamic_cleanup = True
                time.sleep(0.02)
            code = child.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            for state in (work / "logs_dir").glob("pipeline_*"):
                (state / "pipeline.stop").touch()
            child.wait(timeout=45)
            raise
    output = work / "annotated_buffer/device_923001"
    videos = sorted((output / "segments").glob("[0-9]*.mp4"))
    total = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        try:
            assert cap.isOpened(), video
            assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 0.02, video
            decoded = 0
            while cap.read()[0]:
                decoded += 1
            assert decoded > 0, video
            total += decoded
        finally:
            cap.release()
    reports = list((work / "logs_dir").glob("pipeline_*/summary.json"))
    assert code == 0, log_path.read_text(encoding="utf-8")[-6000:]
    assert total == count, (total, count)
    if args.buffer_mode == 'runtime':
        assert observed_dynamic_cleanup, 'cleanup only observed after pipeline exit'
        assert max_pending <= cfg['global'].get('max_buffer_seconds', 8), max_pending
    assert len(videos) == __import__("math").ceil(count / fps), (len(videos), count / fps)
    raw_dirs = list((work / "capture_buffer/device_923001").glob("[0-9]*"))
    if args.buffer_mode == "test":
        assert len(raw_dirs) == len(videos)
        assert sum(len(list(directory.glob("*.jpg"))) for directory in raw_dirs) == count
    else:
        assert not raw_dirs
    for name in ("buffer_101", "buffer_201"):
        protocols = list((work / name).rglob("*.bin"))
        assert len(protocols) == (0 if args.buffer_mode == "runtime" else len(videos))
        if name == "buffer_101":
            assert len(list((work / name).rglob("*.frames.json"))) == (0 if args.buffer_mode == "runtime" else len(videos))
    assert not list(output.glob("*.tmp.mp4"))
    full = list(output.glob("*_full.mp4"))
    assert len(full) == 1, full
    cap = cv2.VideoCapture(str(full[0]))
    decoded_full = 0
    while cap.read()[0]:
        decoded_full += 1
    cap.release()
    assert decoded_full == count, (decoded_full, count)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    lines = log_path.read_text(encoding="utf-8").splitlines()
    first_output = next(i for i, line in enumerate(lines) if "秒段完成" in line)
    capture_end = next(i for i, line in enumerate(lines) if "采集结束:" in line)
    result = {"input_frames": count, "decoded_frames": total, "segments": len(videos),
              "buffer_mode": args.buffer_mode, "dynamic_cleanup_observed": observed_dynamic_cleanup,
              "max_capture_seconds_on_disk": max_pending,
              "fps": fps, "produced_before_capture_end": first_output < capture_end,
              "summary": report, "log": str(log_path)}
    (work / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
