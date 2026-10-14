"""不加载模型的磁盘协议/异常冒烟，所有数据置于独立临时目录。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from InsightfaceModel import protocol_encoder as enc
from InsightfaceModel.model_processor import ModelProcessor
from VideoTag.segment_processor import SegmentProcessor
from VideoTag.protocol_decoder import ProtocolError


def main():
    work = Path(tempfile.mkdtemp(prefix="pipeline_contract_", dir=ROOT / "logs"))
    cfg = {"global": {key: str(work / key) for key in
                       ("capture_buffer", "buffer_101", "buffer_201", "annotated_buffer")},
           "inference": {"fps": 10, "idle_timeout": 1, "face_sim_threshold": 0.6},
           "tag": {"segment_seconds": 60, "fourcc": "mp4v", "follow_poll_interval": 0.01}}
    for key in cfg["global"]:
        (Path(cfg["global"][key]) / "device_4").mkdir(parents=True)
    def seed(ts, count=5, corrupt=False, persons=None):
        directory = Path(cfg["global"]["capture_buffer"]) / "device_4" / str(ts)
        directory.mkdir()
        for index in range(count):
            frame = np.full((120, 160, 3), 40 + index, dtype=np.uint8)
            ok, data = cv2.imencode(".jpg", frame)
            assert ok
            data.tofile(str(directory / f"frame_{index:06d}.jpg"))
        if corrupt:
            (directory / "frame_000000.jpg").write_bytes(b"broken jpeg")
        (directory / "_ready.json").write_text(json.dumps({"frames": count, "fps": 25}), encoding="utf-8")
        enc.write_atomic(Path(cfg["global"]["buffer_101"]) / "device_4" / f"{ts}.bin", enc.encode_101(4, ts, persons or []))
        enc.write_atomic(Path(cfg["global"]["buffer_201"]) / "device_4" / f"{ts}.bin", enc.encode_201(4, ts, []))
        return directory
    tag = SegmentProcessor(4, "", cfg, follow=True)
    stop = threading.Event()
    cases = []
    raw = seed(100, corrupt=True)
    assert tag._annotate_second(100, stop)
    cap = cv2.VideoCapture(str(tag.segments_dir / "100.mp4"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 4 and cap.get(cv2.CAP_PROP_FPS) == 25
    cap.release()
    assert not raw.exists()
    cases.append("empty_detection_and_corrupt_frame")
    raw = seed(101)
    protocol = Path(cfg["global"]["buffer_201"]) / "device_4/101.bin"
    data = bytearray(protocol.read_bytes())
    data[12] ^= 1
    protocol.write_bytes(data)
    try:
        tag._annotate_second(101, stop)
        raise AssertionError("坏 CRC 应被拒绝")
    except ProtocolError as exc:
        assert "CRC" in str(exc), exc
    assert raw.exists() and protocol.exists()
    cases.append("bad_crc_preserves_inputs")
    raw = seed(102)
    stop.set()
    try:
        tag._annotate_second(102, stop)
        raise AssertionError("停止信号应中断秒段")
    except InterruptedError:
        pass
    assert raw.exists() and not (tag.segments_dir / "102.tmp.mp4").exists()
    stop.clear()
    cases.append("cancel_keeps_inputs_and_removes_temporary")
    seed(103, persons=[{"track_id": 7, "xyxy": (20, 30, 50, 80), "conf": 0.9}])
    assert tag._annotate_second(103, stop)
    original = (tag.segments_dir / "103.mp4").read_bytes()
    seed(103, persons=[{"track_id": 7, "xyxy": (20, 30, 50, 80), "conf": 0.9}])
    assert not tag._annotate_second(103, stop)
    assert original == (tag.segments_dir / "103.mp4").read_bytes() and tag._cum_count == 7
    cases.append("idempotent_replay_and_total")
    raw = seed(104)
    (tag.segments_dir / "104.mp4").write_bytes(b"invalid")
    try:
        tag._annotate_second(104, stop)
        raise AssertionError("损坏已有视频应被拒绝")
    except IOError:
        pass
    assert raw.exists()
    cases.append("invalid_existing_output_preserves_inputs")
    model = ModelProcessor(4, cfg)
    raw = seed(105)
    (raw / "_ready.json").unlink()
    (raw / "_writing").touch()
    assert 105 not in model._scan_new_timestamps()
    (raw / "_writing").unlink()
    (raw / "_ready.json").write_text("{}", encoding="utf-8")
    assert 105 in model._scan_new_timestamps()
    cases.append("slow_capture_completion_barrier")
    first = np.eye(1, 512, 0, dtype=np.float32)[0]
    different = np.eye(1, 512, 1, dtype=np.float32)[0]
    assert model._identify_face(first) == model._identify_face(first * 10) == 1
    assert model._identify_face(different) == 2
    cases.append("normalized_cosine_identity_dedup")
    # 正常秒、缺一侧秒与坏 CRC 秒混合：排空成功/跳过/失败计数必须正确。
    state = work / "control"
    state.mkdir()
    (state / "inference.done").write_text("{}", encoding="utf-8")
    os.environ["AIOT_PIPELINE_STATE"] = str(state)
    os.environ["AIOT_PIPELINE_START_TS"] = "200"
    seed(200)
    raw = seed(201)
    (Path(cfg["global"]["buffer_201"]) / "device_4/201.bin").unlink()
    seed(202)
    tag = SegmentProcessor(4, "", cfg, follow=True)
    try:
        assert tag.run(stop) == 2
        assert tag.skipped == 1 and raw.exists()
        assert json.loads((state / "tag.done").read_text()) == {"success": 2, "skipped": 1, "failed": 0}
    finally:
        os.environ.pop("AIOT_PIPELINE_STATE", None)
        os.environ.pop("AIOT_PIPELINE_START_TS", None)
    cases.append("unpaired_second_and_drain")
    (work / "verification.json").write_text(json.dumps({"passed": cases}, indent=2), encoding="utf-8")
    print(json.dumps({"passed": cases, "work": str(work)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
