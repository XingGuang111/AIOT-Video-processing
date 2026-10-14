# -*- coding: utf-8 -*-
"""Process sampled capture frames and publish paired protocols plus per-frame positions."""
import threading
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from runtime_metrics import Metrics
from capture_buffers import second_directories, ready_seconds
from .logger import logger
from .inference_engine import PersonDetector, FaceEmbedder, nms_faces
from .tracker import Tracker
from . import protocol_encoder as proto


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """兼容中文路径读取 JPEG，损坏/空帧返回 None。

    不用 cv2.imread 的原因：imread 在 Windows 上无法处理非 ASCII（中文）路径，
    会静默返回 None；改为 np.fromfile 读字节流 + imdecode 解码，与
    VideoCapture 写盘侧（imencode + tofile）的处理方式对称。
    返回 None 由调用方跳过该帧，不中断整秒处理。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class ModelProcessor:
    """单设备帧处理器：轮询 capture_buffer，生成 101/201 协议。

    一个实例对应一台设备，生命周期为：构造（轻量）→ run() 主循环
    （期间懒加载模型）→ 空闲/被 stop 时 _release() 释放模型。
    """

    def __init__(self, device_no: int, config: Dict, fps: Optional[int] = None):
        """初始化处理器（不加载模型、不读帧，纯参数准备）。

        参数:
            device_no: 设备号（int），决定读写目录 device_N。
            config:    全局配置（config/config.yaml），用到 global 和 inference 两段。
            fps:       抽帧帧率（每秒目录内等间隔取多少帧做推理）；
                       None 时读配置 inference.fps，与采集 FPS 独立；
                       协议时间轴由秒目录决定。
        """
        self.device_no = device_no
        g = config["global"]
        inf = config["inference"]
        self.fps = fps or int(inf["fps"])                  # 命令行 --fps 优先于配置
        self.face_fps = min(self.fps, max(1, int(inf.get("face_fps", self.fps))))
        self.idle_timeout = float(inf["idle_timeout"])     # 空闲回收阈值（秒）

        # ---- 输入/输出目录（只读输入，写输出；目录由使用方保证存在）----
        self.capture_dir = Path(g["capture_buffer"]) / f"device_{device_no}"   # 帧输入
        self.buffer_101 = Path(g["buffer_101"]) / f"device_{device_no}"        # 人体协议输出
        self.buffer_201 = Path(g["buffer_201"]) / f"device_{device_no}"        # 人脸协议输出

        # ---- 模型相关：暂存配置，真正的加载推迟到 _ensure_models() ----
        self._gpu = bool(g.get("gpu", True))
        self._inf = inf
        self.detector: Optional[PersonDetector] = None   # YOLO 人体检测 + ByteTrack
        self.embedder: Optional[FaceEmbedder] = None     # InsightFace 人脸检测+特征
        self.tracker: Optional[Tracker] = None           # ByteTrack 原始 id → 连续编号

        # 已处理过的秒级时间戳集合：防止重复处理同一秒目录
        # （内存态，进程重启后从零开始，重复处理也只是覆盖同名 .bin，无副作用）
        self._processed_ts: set = set()
        self.pipeline_state = Path(os.environ["AIOT_PIPELINE_STATE"]) if os.environ.get("AIOT_PIPELINE_STATE") else None
        self.min_ts = int(os.environ.get("AIOT_PIPELINE_START_TS", "0"))
        self.metrics = Metrics("inference")
        self.failed_seconds = 0
        self._face_gallery: List[np.ndarray] = []
        self.face_sim_threshold = float(inf.get("face_sim_threshold", 0.6))

    # ---- 资源管理 ----
    def _ensure_models(self) -> None:
        """懒加载模型：首次有帧可处理时才创建 detector/embedder/tracker。

        好处：启动推理服务时不占资源；空闲设备永远不会触发加载；
        空闲回收后再次来帧时自动重新加载（detector 为 None 即触发）。
        """
        if self.detector is not None:
            return  # 已加载过，直接返回（幂等）
        inf = self._inf
        # YOLO 人体检测器：conf/过滤阈值/跟踪配置全部来自 config.yaml
        self.detector = PersonDetector(
            model_path=inf["yolo_model"], conf=float(inf["yolo_conf"]), gpu=self._gpu,
            tracker=inf.get("tracker", "bytetrack.yaml"),
            min_pixels=int(inf.get("person_min_pixels", 0)),
            edge_margin=int(inf.get("edge_margin", 0)),
            min_top_margin=int(inf.get("person_min_top_margin", 0)),
            half=bool(inf.get("yolo_half", True)),
        )
        # InsightFace 人脸特征器：buffalo_l 模型包（检测 + 512 维特征）
        self.embedder = FaceEmbedder(
            gpu=self._gpu, det_size=inf.get("face_det_size", (640, 640)),
            root=inf.get("face_root") or None,
        )
        # 连续人数编号器（纯内存记账，无模型权重）
        self.tracker = Tracker()

    def _release(self) -> None:
        """释放模型资源并复位跟踪状态（任务结束/空闲回收时调用）。

        置 None 后引用计数归零，Python 会回收显存/内存；
        tracker.reset() 清空连续编号，保证下次启动编号重新从 1 开始。
        """
        self.detector = None
        self.embedder = None
        if self.tracker:
            self.tracker.reset()
        logger.info("[设备%d] 已释放模型资源", self.device_no)

    # ---- 主循环 ----
    def run(self, stop_event: threading.Event) -> None:
        try:
            self._run_loop(stop_event)
        finally:
            self.metrics.publish(finished=True)
            self._release()

    def _run_loop(self, stop_event: threading.Event) -> None:
        """推理主循环：轮询 capture_buffer，逐秒处理，直到停止或空闲超时。

        参数:
            stop_event: 外部停止信号（由 InferenceWorker.stop() 置位）。
                        循环开头和每轮空闲等待都检查它，保证 1 秒内响应停止。

        退出路径有三条：stop_event 置位 / 空闲超 idle_timeout 秒 / 未捕获异常
        （异常由 InferenceWorker.run 的 try 兜底记录到 worker.error）。
        """
        logger.info("[设备%d] 推理处理器启动，轮询: %s", self.device_no, self.capture_dir)
        last_active = time.time()  # 最近一次"有帧可处理"的时间，用于空闲判定
        while not stop_event.is_set():
            if self.pipeline_state and (self.pipeline_state / "inference.stop").exists():
                break
            # 先读结束标记，再扫描，避免尾秒刚发布时用旧扫描结果提前退出。
            capture_done = bool(self.pipeline_state and (self.pipeline_state / "capture.done").exists())
            new_ts = self._scan_new_timestamps()
            if not new_ts:
                # ---- 无新帧：空闲分支 ----
                if capture_done:
                    break  # 采集已完成，所有 ready 秒均已处理；无需等待 idle_timeout。
                if not self.pipeline_state and time.time() - last_active >= self.idle_timeout:
                    logger.info("[设备%d] 空闲 %.0fs 无新帧，自动回收", self.device_no, self.idle_timeout)
                    break
                # 用 wait 而非 sleep：stop() 置位会立刻唤醒，停止响应不延迟
                stop_event.wait(0.1)
                continue

            # ---- 有新帧：先确保模型已加载（懒加载），再逐秒处理 ----
            if self.detector is None:
                with self.metrics.measure("model_loading"):
                    self._ensure_models()
            if not self.metrics.data.get("device"):
                self.metrics.data["device"] = str(self.detector.device)
                self.metrics.data["face_providers"] = self.embedder.providers
                self.metrics.publish()
            for ts in new_ts:
                if stop_event.is_set():
                    break  # 本批秒目录还没处理完也立即停，剩余秒下次再说
                try:
                    with self.metrics.measure("inference_total"):
                        self._process_second(ts)
                    self.metrics.data["processed"] += 1
                    self.metrics.publish()
                except Exception as e:
                    # 单秒异常隔离：记日志后继续处理后续秒，不拖死整个任务
                    logger.error("[设备%d] 处理时间戳 %d 异常: %s", self.device_no, ts, e)
                    self.failed_seconds += 1
                # 无论成功失败都登记：失败的秒不反复重试（数据本身有问题时重试无意义）
                self._processed_ts.add(ts)
            last_active = time.time()  # 有活干就刷新活跃时间，重置空闲计时
        if self.pipeline_state and not stop_event.is_set() and not (self.pipeline_state / "inference.stop").exists():
            proto.write_atomic(self.pipeline_state / "inference.done", json.dumps({
                "seconds": len(self._processed_ts), "failed": self.failed_seconds}).encode("utf-8"))
        logger.info("[设备%d] 推理处理器已停止", self.device_no)

    def _scan_new_timestamps(self) -> List[int]:
        """返回尚未处理、且非当前秒（已写完）的时间戳目录，升序。

        跳过当前秒的原因：VideoCapture 按真实时间建秒目录并持续写帧，
        当前秒的目录可能还在写入中，此时处理会漏掉后半秒的帧；
        等它"翻篇"成历史秒后再处理，目录内容就是完整稳定的。
        升序返回保证 ByteTrack 按时间顺序跟踪（乱序会导致 id 断裂）。
        """
        if not self.capture_dir.exists():
            return []
        if self.pipeline_state:
            return sorted(int(directory.name) for directory in ready_seconds(self.capture_dir, self.min_ts)
                          if int(directory.name) not in self._processed_ts)
        now_sec = int(time.time())
        result = []
        for d in second_directories(self.capture_dir, self.min_ts):
            ts = int(d.name)
            if ts in self._processed_ts or ts < self.min_ts:
                continue  # 跳过已处理的和可能仍在写入的当前秒
            if (d / "_writing").exists():
                continue
            if not (d / "_ready.json").is_file() and ts >= now_sec:
                continue
            result.append(ts)
        return sorted(result)

    def _process_second(self, ts: int) -> None:
        """处理一个秒目录：该秒内的所有目标聚合成 1 个 101 + 1 个 201 协议帧。

        处理流水线（顺序即数据流）：
            抽帧 → 逐帧(人体检测+跟踪编号 / 人脸检测+帧内NMS+关联人体)
            → 跨帧 NMS → 按 person_id 聚合 → 编 face_id → 原子写盘。
        """
        ts_dir = self.capture_dir / str(ts)
        frames = sorted(ts_dir.glob("*.jpg"))  # 按文件名（帧序号）升序
        if not frames:
            return
        source_frames = len(frames)
        frames = self._sample(frames)
        face_indices = set(np.linspace(0, len(frames) - 1, min(self.face_fps, len(frames))).astype(int))

        persons_agg: Dict[int, Dict] = {}  # track_id -> 人体记录（该秒内取置信度最高的一次）
        faces_per_frame: List[Dict] = []   # 各帧检出的人脸暂存，最后统一做跨帧聚合
        frame_annotations = []

        # ========== 第一阶段：逐帧检测 ==========
        for frame_index, fp in enumerate(frames):
            with self.metrics.measure("image_read"):
                img = _imread_unicode(fp)
            if img is None:
                logger.warning("[设备%d] 跳过损坏帧: %s", self.device_no, fp.name)
                continue
            # ---- 人体：YOLO 检测 + ByteTrack 跟踪（detect 内部 persist 保持跟踪状态）----
            with self.metrics.measure("yolo"):
                persons = self.detector.detect(img)
            # 把 ByteTrack 原始 id 原地改写为从 1 递增的连续人数编号
            with self.metrics.measure("tracking"):
                self.tracker.update(persons, now=ts + int(fp.stem.rsplit("_", 1)[-1]) / max(1, source_frames))
            # 人体去重：同一 track_id 在该秒多帧中可能出现多次，留置信度最高的
            for p in persons:
                tid = p["track_id"]
                if tid not in persons_agg or p["conf"] > persons_agg[tid]["conf"]:
                    persons_agg[tid] = p
            # ---- 人脸：整帧检测 + 帧内 NMS（去同帧冗余框），再暂存待跨帧聚合 ----
            # 人脸在本秒中均匀抽样；只关联同一帧人体，避免复用旧框造成身份错配。
            frame_faces = []
            if frame_index in face_indices:
                with self.metrics.measure("face"):
                    frame_faces = nms_faces(self.embedder.embed(img), iou_thr=0.5)
            self.metrics.data["frames"] += 1
            sample_faces = []
            for fc in frame_faces:
                fx1, fy1, fx2, fy2 = fc["xyxy"]
                record = {
                    "xyxy": (fx1, fy1, fx2, fy2),
                    "conf": fc["conf"],
                    "embedding": fc["embedding"],
                    # 人脸框中心点落入哪个人体框，就归属哪个人（track_id）
                    "person_id": self._match_person(fx1, fy1, fx2, fy2, persons),
                }
                faces_per_frame.append(record)
                sample_faces.append(record)
            frame_annotations.append({"index": int(fp.stem.rsplit('_', 1)[-1]),
                                      "persons": [dict(person) for person in persons], "faces": sample_faces})

        # ========== 第二阶段：秒内聚合（三层去重漏斗的后两层）==========
        # 跨帧 NMS：同秒内多帧检测到的同一张脸（IoU>0.3）合并为一个，保留 conf 最高
        # 这是修复"一张脸画 18 个红框"bug 的关键：之前用 embedding.tobytes() 去重，
        # 但同脸跨帧的 embedding 必然不同（光照/角度变化），导致同秒内 faces 数量 = 帧数 × 单帧脸数。
        with self.metrics.measure("aggregation"):
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
                    # 未关联到人体的脸（如画面边缘只有脸没有人身）不参与聚合，全部保留
                    no_person_faces.append(f)
                elif pid not in best_per_person or f["conf"] > best_per_person[pid]["conf"]:
                    best_per_person[pid] = f
            faces_agg = list(best_per_person.values()) + no_person_faces
            faces_agg.sort(key=lambda f: f["xyxy"][0])
            for face_index, face in enumerate(faces_agg):
                face["face_id"] = face_index
            for sample in frame_annotations:
                usable_faces = []
                for face_index, face in enumerate(sorted(sample["faces"], key=lambda f: f["xyxy"][0])):
                    usable_faces.append({"xyxy": list(face["xyxy"]), "face_id": face_index,
                                         "tracking_id": self._identify_face(face["embedding"]),
                                         "person_id": int(face["person_id"]), "conf": float(face["conf"])})
                sample["faces"] = usable_faces


        # ========== 第三阶段：编码落盘（原子写，下游不会读到半成品）==========
        with self.metrics.measure("protocol_write"):
            # Publish the per-frame positions before the paired protocols announce readiness.
            proto.write_atomic(self.buffer_101 / f"{ts}.frames.json",
                               json.dumps({"ts": ts, "frames": frame_annotations}).encode('utf-8'))
            person_list = list(persons_agg.values())
            proto.write_atomic(self.buffer_101 / f"{ts}.bin",
                                proto.encode_101(self.device_no, ts, person_list))
            proto.write_atomic(self.buffer_201 / f"{ts}.bin",
                                proto.encode_201(self.device_no, ts, faces_agg))
        logger.info("[设备%d] ts=%d 人体=%d 人脸=%d", self.device_no, ts, len(person_list), len(faces_agg))

    def _sample(self, frames: List[Path]) -> List[Path]:
        """秒目录内按 fps 等间隔抽帧。

        采集端是全量写帧（如 18/30/60 帧每秒），推理端不需要也不能逐帧处理
        （算力不够且无必要——1 秒内目标位置变化有限）。用 linspace 在
        [0, n-1] 上均匀取 fps 个下标，保证抽到的时间点在整秒内均匀分布；
        set 去重是防 n 略大于 fps 时 linspace 取整产生重复下标。
        """
        n = len(frames)
        if n <= self.fps:
            return frames  # 帧数不足 fps 时全用（采集降采样或秒尾目录的常见情况）
        idx = np.linspace(0, n - 1, self.fps).astype(int)
        return [frames[i] for i in sorted(set(idx))]

    def _identify_face(self, embedding) -> int:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 512 or not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("无效 InsightFace 特征")
        vector = vector / norm
        if self._face_gallery:
            scores = np.asarray(self._face_gallery) @ vector
            index = int(np.argmax(scores))
            if float(scores[index]) > self.face_sim_threshold:
                return index + 1
        self._face_gallery.append(vector.copy())
        return len(self._face_gallery)

    @staticmethod
    def _match_person(fx1: int, fy1: int, fx2: int, fy2: int, persons: List[Dict]) -> int:
        """人脸框中心落在哪个人体框内即取其 track_id，无匹配返回 NO_PERSON_ID。

        用"中心点包含"而非 IoU：人脸框通常完全在人体框内部（头部区域），
        中心点判定简单可靠；人脸框很小，用 IoU 反而容易因框抖动失配。
        track_id >= 0 的过滤是排除 ByteTrack 偶发输出的无效 id（-1）。
        多个人体框重叠时取第一个命中的（ persons 顺序由检测器给出）。
        """
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        for p in persons:
            x1, y1, x2, y2 = p["xyxy"]
            if x1 <= cx <= x2 and y1 <= cy <= y2 and p["track_id"] >= 0:
                return p["track_id"]
        return proto.NO_PERSON_ID
