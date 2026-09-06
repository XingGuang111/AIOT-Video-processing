# -*- coding: utf-8 -*-
"""数据层：101/201 二进制协议编码、CRC32 校验、原子写入。

协议格式（统一 Big-Endian）：
    start_magic(4B) | total_length(>I) | protocol_no(>H) | device_no(>I)
    | unix_ts(>Q) | record_count(>H) | body | crc32(>I) | end_magic(4B)
CRC32 覆盖范围：start_magic .. body（不含 crc32 与 end_magic）。
"""
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---- 协议常量（编码/解码端必须一致）----
START_MAGIC = b"\xAA\x55\xAA\x55"
END_MAGIC = b"\x55\xAA\x55\xAA"
PROTO_101 = 101
PROTO_201 = 201
NO_PERSON_ID = 0xFFFFFFFF  # 人脸未关联人体时的占位 person_id

# 帧头（紧跟 start_magic 之后）：total_length, protocol_no, device_no, unix_ts, record_count
_HEADER_FMT = ">IHIQH"
# 101 记录：target_id, x1, y1, x2, y2, confidence, class_id
_REC_101_FMT = ">Iiiiif H".replace(" ", "")
# 201 记录：person_id, x1, y1, x2, y2, face_id, face_confidence, 512×float
_REC_201_FMT = ">Iiiii I f 512f".replace(" ", "")

REC_101_SIZE = struct.calcsize(_REC_101_FMT)
REC_201_SIZE = struct.calcsize(_REC_201_FMT)


def _frame(protocol_no: int, device_no: int, ts: int, body: bytes, record_count: int) -> bytes:
    """组装完整协议帧。"""
    # total_length = start_magic + header + body + crc32 + end_magic
    total_length = len(START_MAGIC) + struct.calcsize(_HEADER_FMT) + len(body) + 4 + len(END_MAGIC)
    header = struct.pack(_HEADER_FMT, total_length, protocol_no, device_no, ts, record_count)
    payload = START_MAGIC + header + body
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", crc) + END_MAGIC


def encode_101(device_no: int, ts: int, persons: List[Dict]) -> bytes:
    """编码人体检测协议。

    persons: [{"track_id": int, "xyxy": (x1,y1,x2,y2), "conf": float}]
    """
    body = bytearray()
    for p in persons:
        x1, y1, x2, y2 = (int(v) for v in p["xyxy"])
        body += struct.pack(
            _REC_101_FMT,
            int(p["track_id"]) & 0xFFFFFFFF,
            x1, y1, x2, y2,
            float(p["conf"]),
            0,  # class_id 恒为 0 (person)
        )
    return _frame(PROTO_101, device_no, ts, bytes(body), len(persons))


def encode_201(device_no: int, ts: int, faces: List[Dict]) -> bytes:
    """编码人脸特征协议。

    faces: [{"person_id": int, "xyxy": (x1,y1,x2,y2), "face_id": int,
             "conf": float, "embedding": np.ndarray(512,)}]
    """
    body = bytearray()
    for fc in faces:
        x1, y1, x2, y2 = (int(v) for v in fc["xyxy"])
        emb = np.asarray(fc["embedding"], dtype=np.float32).flatten()
        if emb.size != 512:
            raise ValueError(f"人脸特征向量维度必须为 512，实际 {emb.size}")
        body += struct.pack(
            _REC_201_FMT,
            int(fc["person_id"]) & 0xFFFFFFFF,
            x1, y1, x2, y2,
            int(fc["face_id"]) & 0xFFFFFFFF,
            float(fc["conf"]),
            *emb.tolist(),
        )
    return _frame(PROTO_201, device_no, ts, bytes(body), len(faces))


def write_atomic(path: Path, data: bytes) -> None:
    """原子写入：先写 .tmp，再 os.replace 重命名，避免下游读到半成品。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
