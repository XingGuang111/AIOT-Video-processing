# -*- coding: utf-8 -*-
"""引擎层：帧处理主流程（采样 → 人体检测 → 人脸特征 → 跟踪 → 协议落盘）。"""
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .logger import logger
from .inference_engine import PersonDetector, FaceEmbedder, nms_faces
from .tracker import Tracker
from . import protocol_encoder as proto


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """兼容中文路径读取 JPEG，损坏/空帧返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class ModelProcessor:
    """单设备帧处理器：轮询 capture_buffer，生成 101/201 协议。"""

    def __init__(self, device_no: int, config: Dict, fps: Optional[int] = None):
        self.device_no = device_no
        g = config["global"]
        inf = config["inference"]
        self.fps = fps or int(inf["fps"])
        self.idle_timeout = float(inf["idle_timeout"])

        self.capture_dir = Path(g["capture_buffer"]) / f"device_{device_no}"
        self.buffer_101 = Path(g["buffer_101"]) / f"device_{device_no}"
        self.buffer_201 = Path(g["buffer_201"]) / f"device_{device_no}"

        self._cfg = config
        self._gpu = bool(g.get("gpu", True))
        self._inf = inf
        self.detector: Optional[PersonDetector] = None
        self.embedder: Optional[FaceEmbedder] = None
        self.tracker: Optional[Tracker] = None

        self._processed_ts: set = set()

    # ---- 资源管理 ----
    def _ensure_models(self) -> None:
        if self.detector is not None:
            return
        inf = self._inf
        self.detector = PersonDetector(
            model_path=inf["yolo_model"], conf=float(inf["yolo_conf"]), gpu=self._gpu,
            tracker=inf.get("tracker", "bytetrack.yaml"),
            min_pixels=int(inf.get("person_min_pixels", 0)),
            edge_margin=int(inf.get("edge_margin", 0)),
            min_top_margin=int(inf.get("person_min_top_margin", 0)),
        )
        self.embedder = FaceEmbedder(
            gpu=self._gpu, det_size=inf.get("face_det_size", (640, 640)),
            root=inf.get("face_root") or None,
        )
        self.tracker = Tracker(tracker_cfg=inf.get("tracker", "bytetrack.yaml"))

    def _release(self) -> None:
        self.detector = None
        self.embedder = None
        if self.tracker:
            self.tracker.reset()
        logger.info("[设备%d] 已释放模型资源", self.device_no)

    # ---- 主循环 ----
    def run(self, stop_event: threading.Event) -> None:
        logger.info("[设备%d] 推理处理器启动，轮询: %s", self.device_no, self.capture_dir)
        last_active = time.time()
        while not stop_event.is_set():
            new_ts = self._scan_new_timestamps()
            if not new_ts:
                if time.time() - last_active >= self.idle_timeout:
                    logger.info("[设备%d] 空闲 %.0fs 无新帧，自动回收", self.device_no, self.idle_timeout)
                    break
                stop_event.wait(1.0)
                continue

            self._ensure_models()
            for ts in new_ts:
                if stop_event.is_set():
                    break
                try:
                    self._process_second(ts)
                except Exception as e:
                    logger.error("[设备%d] 处理时间戳 %d 异常: %s", self.device_no, ts, e)
                self._processed_ts.add(ts)
            last_active = time.time()
        self._release()
        logger.info("[设备%d] 推理处理器已停止", self.device_no)

    def _scan_new_timestamps(self) -> List[int]:
        """返回尚未处理、且非当前秒（已写完）的时间戳目录，升序。"""
        if not self.capture_dir.exists():
            return []
        now_sec = int(time.time())
        result = []
        for d in self.capture_dir.iterdir():
            if not d.is_dir() or not d.name.isdigit():
                continue
            ts = int(d.name)
            if ts in self._processed_ts or ts >= now_sec:
                continue  # 跳过可能仍在写入的当前秒
            result.append(ts)
        return sorted(result)

    def _process_second(self, ts: int) -> None:
        ts_dir = self.capture_dir / str(ts)
        frames = sorted(ts_dir.glob("*.jpg"))
        if not frames:
            return
        frames = self._sample(frames)

        persons_agg: Dict[int, Dict] = {}  # track_id -> 人体记录（取置信度最高）
        faces_per_frame: List[Dict] = []   # 每帧人脸检测结果，最后做 NMS 聚合

        for fp in frames:
            img = _imread_unicode(fp)
            if img is None:
                logger.warning("[设备%d] 跳过损坏帧: %s", self.device_no, fp.name)
                continue
            persons = self.detector.detect(img)
            self.tracker.update(persons)
            # 人体去重：每个 track_id 留置信度最高的
            for p in persons:
                tid = p["track_id"]
                if tid not in persons_agg or p["conf"] > persons_agg[tid]["conf"]:
                    persons_agg[tid] = p
            # 每帧都跑人脸检测；先做帧内 NMS（去同帧冗余框），再加入跨帧聚合列表
            frame_faces = nms_faces(self.embedder.embed(img), iou_thr=0.5)
            for fc in frame_faces:
                fx1, fy1, fx2, fy2 = fc["xyxy"]
                faces_per_frame.append({
                    "xyxy": (fx1, fy1, fx2, fy2),
                    "conf": fc["conf"],
                    "embedding": fc["embedding"],
                    "person_id": self._match_person(fx1, fy1, fx2, fy2, persons),
                })

        # 跨帧 NMS：同秒内多帧检测到的同一张脸（IoU>0.3）合并为一个，保留 conf 最高
        # 这是修复"一张脸画 18 个红框"bug 的关键：之前用 embedding.tobytes() 去重，
        # 但同脸跨帧的 embedding 必然不同（光照/角度变化），导致同秒内 faces 数量 = 帧数 × 单帧脸数。
        faces_agg = nms_faces(faces_per_frame, iou_thr=0.3)
        # 按 person_id 聚合：每人在该秒内只保留 conf 最高的 1 张脸。
        # 这是修复"同一张脸画 5 个红框"bug 的关键：InsightFace 在低质量/复杂背景下常把
        # 头部/帽/肩膀/衣服纹理误检为多张脸（IoU 都不高无法用 NMS 合并），
        # 但它们都关联到同一 person_id，按 person_id 聚合即可确保"每人 1 个红框"。
        best_per_person: Dict[int, Dict] = {}
        no_person_faces: List[Dict] = []
        for f in faces_agg:
            pid = f["person_id"]
            if pid == proto.NO_PERSON_ID:
                no_person_faces.append(f)
            elif pid not in best_per_person or f["conf"] > best_per_person[pid]["conf"]:
                best_per_person[pid] = f
        faces_agg = list(best_per_person.values()) + no_person_faces
        # face_id 字段：与 person_id 同值（=绑定的 track_id），让红框 ID 跟随人稳定
        for f in faces_agg:
            f["face_id"] = f["person_id"]

        person_list = list(persons_agg.values())
        proto.write_atomic(self.buffer_101 / f"{ts}.bin",
                            proto.encode_101(self.device_no, ts, person_list))
        proto.write_atomic(self.buffer_201 / f"{ts}.bin",
                            proto.encode_201(self.device_no, ts, faces_agg))
        logger.info("[设备%d] ts=%d 人体=%d 人脸=%d", self.device_no, ts, len(person_list), len(faces_agg))

    def _sample(self, frames: List[Path]) -> List[Path]:
        """秒目录内按 fps 等间隔抽帧。"""
        n = len(frames)
        if n <= self.fps:
            return frames
        idx = np.linspace(0, n - 1, self.fps).astype(int)
        return [frames[i] for i in sorted(set(idx))]

    def _faces_in_frame(self, img: np.ndarray, persons: List[Dict]) -> List[Dict]:
        """对整帧检测人脸：face_id 为整帧人脸数组的统一下标（0,1,2…）。

        每张人脸按"框中心落在哪个人体框内"关联该人体 track_id，无则置 NO_PERSON_ID。
        """
        out = []
        for face_id, fc in enumerate(self.embedder.embed(img)):
            fx1, fy1, fx2, fy2 = fc["xyxy"]
            out.append({
                "person_id": self._match_person(fx1, fy1, fx2, fy2, persons),
                "xyxy": (fx1, fy1, fx2, fy2),
                "face_id": face_id,
                "conf": fc["conf"],
                "embedding": fc["embedding"],
            })
        return out

    @staticmethod
    def _match_person(fx1: int, fy1: int, fx2: int, fy2: int, persons: List[Dict]) -> int:
        """人脸框中心落在哪个人体框内即取其 track_id，无匹配返回 NO_PERSON_ID。"""
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        for p in persons:
            x1, y1, x2, y2 = p["xyxy"]
            if x1 <= cx <= x2 and y1 <= cy <= y2 and p["track_id"] >= 0:
                return p["track_id"]
        return proto.NO_PERSON_ID
