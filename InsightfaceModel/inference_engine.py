# -*- coding: utf-8 -*-
"""基础层：YOLOv8 人体检测 与 InsightFace 人脸特征提取底层封装。

- PersonDetector：仅保留 person 类（COCO id=0），内置 ByteTrack 输出稳定 track_id。
- FaceEmbedder：buffalo_l 模型，输出人脸框、置信度、512 维特征。
两者均为薄封装，权重首次运行自动下载。
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

import numpy as np

from .logger import logger

PERSON_CLASS_ID = 0  # COCO person


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_faces(faces: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
    """对同帧内的人脸框做 NMS：IoU > 阈值的多框合并为一个，保留 conf 最高的。

    InsightFace SCRFD 在某些角度/清晰度下可能对同一张脸输出多个高度重叠的框，
    合并后保证同张脸在单帧只出一个框。
    """
    if len(faces) <= 1:
        return faces
    # 按 conf 降序
    order = sorted(range(len(faces)), key=lambda i: faces[i]["conf"], reverse=True)
    keep: List[int] = []
    suppressed = [False] * len(faces)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if _iou(faces[i]["xyxy"], faces[j]["xyxy"]) > iou_thr:
                suppressed[j] = True
    return [faces[i] for i in keep]


class PersonDetector:
    """YOLOv8 人体检测 + ByteTrack 跟踪。"""

    def __init__(self, model_path: str, conf: float = 0.25, gpu: bool = True,
                 tracker: str = "bytetrack.yaml", min_pixels: int = 0, edge_margin: int = 0,
                 min_top_margin: int = 0, half: bool = True):
        from ultralytics import YOLO  # 延迟导入，避免无依赖时模块导入即失败
        import torch
        import cv2

        self.conf = conf
        self.tracker = tracker
        self.min_pixels = min_pixels
        self.edge_margin = edge_margin
        # 顶部贴边过滤：人体框顶部距画面顶部 < 此值视为"头部被裁掉（只剩下半身）"，
        # 直接丢弃，避免分配 track_id 后画面出现"只有绿框无 F 框"的鬼影
        self.min_top_margin = min_top_margin
        # 自动探测 CUDA，不可用则降级 CPU
        use_gpu = gpu and torch.cuda.is_available()
        if use_gpu:
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
        if gpu and not use_gpu:
            logger.warning("[Engine] 未检测到可用 CUDA，YOLO 自动降级为 CPU")
        self.device = 0 if use_gpu else "cpu"
        self.half = bool(half and use_gpu)
        logger.info("[Engine] 加载 YOLOv8 模型: %s (device=%s)", model_path, self.device)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """对单帧检测人体，返回 [{track_id, xyxy, conf}]。

        按时间顺序连续调用以维持 ByteTrack 状态（persist=True）。
        """
        results = self.model.track(
            frame, classes=[PERSON_CLASS_ID], conf=self.conf,
            persist=True, tracker=self.tracker, device=self.device, half=self.half, verbose=False,
        )
        out: List[Dict] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return out

        h, w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            if not self._valid_box(x1, y1, x2, y2, w, h):
                continue
            track_id = int(ids[i]) if ids is not None else -1
            out.append({"track_id": track_id, "xyxy": (x1, y1, x2, y2), "conf": float(confs[i])})
        return out

    def _valid_box(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
        """最小像素 + 边缘区域 + 顶部贴边（残缺下半身）过滤无效人体框。"""
        if (x2 - x1) < self.min_pixels or (y2 - y1) < self.min_pixels:
            return False
        m = self.edge_margin
        if m > 0 and (x1 <= m and y1 <= m and x2 >= w - m and y2 >= h - m):
            return False
        # 顶部贴边：人体框顶部紧贴画面顶部，说明头部被裁掉只剩下半身，
        # 此类人体不参与跟踪/识别（无脸可检、ID 浪费、显示鬼影）
        if self.min_top_margin > 0 and y1 < self.min_top_margin:
            return False
        return True


class FaceEmbedder:
    """InsightFace buffalo_l：人脸检测 + 512 维 embedding。"""

    def __init__(self, gpu: bool = True, det_size=(640, 640), root: Optional[str] = None):
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.utils.storage import ensure_available
        import onnxruntime as ort

        # 自动探测 onnxruntime 是否有 CUDA provider，不可用则降级 CPU
        use_gpu = gpu and ("CUDAExecutionProvider" in ort.get_available_providers())
        if gpu and not use_gpu:
            logger.warning("[Engine] onnxruntime 无 CUDA provider，InsightFace 自动降级为 CPU")
        providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
                     if use_gpu else ["CPUExecutionProvider"])
        logger.info("[Engine] 加载 InsightFace buffalo_l (gpu=%s)", use_gpu)
        model_dir = Path(ensure_available('models', 'buffalo_l', root=root or '~/.insightface'))
        # Only initialize the two models used, avoiding unused landmark/age sessions.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1 if use_gpu else max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        detection_path = str(model_dir / 'det_10g.onnx')
        recognition_path = str(model_dir / 'w600k_r50.onnx')
        self.detector = SCRFD(model_file=detection_path,
                              session=ort.InferenceSession(detection_path, sess_options=options, providers=providers))
        self.recognizer = ArcFaceONNX(model_file=recognition_path,
                                      session=ort.InferenceSession(recognition_path, sess_options=options, providers=providers))
        self.detector.prepare(ctx_id=0 if use_gpu else -1, input_size=tuple(det_size), det_thresh=0.5)
        self.recognizer.prepare(ctx_id=0 if use_gpu else -1)
        self.providers = self.detector.session.get_providers()

    def embed(self, img_bgr: np.ndarray) -> List[Dict]:
        """检测人脸并提取特征，返回 [{xyxy, conf, embedding(512,)}]。

        InsightFace buffalo_l 输出的 bbox 已是原图坐标，无需额外缩放。
        """
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        from insightface.app.common import Face
        bboxes, landmarks = self.detector.detect(img_bgr, max_num=0, metric='default')
        out: List[Dict] = []
        for index, bbox in enumerate(bboxes):
            if bbox[4] < 0.3:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            bw, bh = x2 - x1, y2 - y1
            # 尺寸过滤：人脸不可能太大（>画面1/3）或太小（<20px）或长宽比离谱
            if bw < 20 or bh < 20 or bw > w / 3 or bh > h / 3 or bw / bh > 2 or bh / bw > 2:
                continue
            f = Face(bbox=bbox[:4], kps=landmarks[index] if landmarks is not None else None,
                     det_score=bbox[4])
            self.recognizer.get(img_bgr, f)
            out.append({
                "xyxy": (x1, y1, x2, y2),
                "conf": float(f.det_score),
                "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
            })
        return out
