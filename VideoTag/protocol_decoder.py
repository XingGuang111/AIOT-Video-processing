# -*- coding: utf-8 -*-
"""数据层：101/201 二进制协议校验、解析，按时间戳重组内存数据。

协议格式与 InsightfaceModel/protocol_encoder.py 严格一致（模块间不互相导入，
按 README「目录解耦、无代码导入依赖」约束各自实现同一格式）。
"""
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .logger import logger

START_MAGIC = b"\xAA\x55\xAA\x55"
END_MAGIC = b"\x55\xAA\x55\xAA"
PROTO_101 = 101
PROTO_201 = 201
NO_PERSON_ID = 0xFFFFFFFF

_HEADER_FMT = ">IHIQH"
_REC_101_FMT = ">Iiiiif H".replace(" ", "")
_REC_201_FMT = ">Iiiii I f 512f".replace(" ", "")
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_REC_101_SIZE = struct.calcsize(_REC_101_FMT)
_REC_201_SIZE = struct.calcsize(_REC_201_FMT)


class ProtocolError(Exception):
    """协议校验/解析异常。"""


def decode(data: bytes) -> Dict:
    """解析单个协议帧，校验失败抛 ProtocolError。

    返回 {"protocol_no", "device_no", "ts", "records": [...]}。
    """
    if len(data) < len(START_MAGIC) + _HEADER_SIZE + 4 + len(END_MAGIC):
        raise ProtocolError("数据长度不足")
    if data[:4] != START_MAGIC:
        raise ProtocolError("起始魔数错误")
    if data[-4:] != END_MAGIC:
        raise ProtocolError("结束魔数错误")

    off = len(START_MAGIC)
    total_length, protocol_no, device_no, ts, record_count = struct.unpack_from(_HEADER_FMT, data, off)
    if total_length != len(data):
        raise ProtocolError(f"总长度不匹配: 头部 {total_length} != 实际 {len(data)}")

    # CRC 校验：覆盖 start_magic..body
    crc_off = len(data) - len(END_MAGIC) - 4
    expected_crc = struct.unpack_from(">I", data, crc_off)[0]
    actual_crc = zlib.crc32(data[:crc_off]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise ProtocolError(f"CRC32 校验失败: {expected_crc} != {actual_crc}")

    body_off = off + _HEADER_SIZE
    if protocol_no == PROTO_101:
        records = _parse_101(data, body_off, record_count)
    elif protocol_no == PROTO_201:
        records = _parse_201(data, body_off, record_count)
    else:
        raise ProtocolError(f"未知协议号: {protocol_no}")

    return {"protocol_no": protocol_no, "device_no": device_no, "ts": ts, "records": records}


def _parse_101(data: bytes, off: int, count: int) -> List[Dict]:
    out = []
    for _ in range(count):
        tid, x1, y1, x2, y2, conf, cls = struct.unpack_from(_REC_101_FMT, data, off)
        off += _REC_101_SIZE
        out.append({"track_id": tid, "xyxy": (x1, y1, x2, y2), "conf": conf, "class_id": cls})
    return out


def _parse_201(data: bytes, off: int, count: int) -> List[Dict]:
    out = []
    for _ in range(count):
        fields = struct.unpack_from(_REC_201_FMT, data, off)
        off += _REC_201_SIZE
        pid, x1, y1, x2, y2, fid, conf = fields[:7]
        emb = np.array(fields[7:], dtype=np.float32)
        out.append({
            "person_id": pid, "xyxy": (x1, y1, x2, y2), "face_id": fid,
            "conf": conf, "embedding": emb,
        })
    return out


def _read_bytes_unicode(path: Path) -> bytes:
    return np.fromfile(str(path), dtype=np.uint8).tobytes()


def load_device(buffer_101: Path, buffer_201: Path, device_no: int) -> Dict[int, Dict]:
    """读取某设备全部 101/201 协议，按 ts 重组。

    返回 {ts: {"persons": [...], "faces": [...]}}；坏文件记 WARN 跳过。
    """
    table: Dict[int, Dict] = {}

    def _ingest(bin_dir: Path, key: str, proto_no: int):
        dev_dir = bin_dir / f"device_{device_no}"
        if not dev_dir.exists():
            return
        for fp in sorted(dev_dir.glob("*.bin")):
            try:
                frame = decode(_read_bytes_unicode(fp))
            except ProtocolError as e:
                logger.warning("[Decoder] 跳过损坏协议 %s: %s", fp.name, e)
                continue
            if frame["protocol_no"] != proto_no:
                continue
            entry = table.setdefault(frame["ts"], {"persons": [], "faces": []})
            entry[key] = frame["records"]

    _ingest(buffer_101, "persons", PROTO_101)
    _ingest(buffer_201, "faces", PROTO_201)
    return table
