# -*- coding: utf-8 -*-
"""配置层：加载并校验根目录 config/config.yaml 全局参数。"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict

# _ROOT：项目根目录。当前文件位于 <根>/InsightfaceModel/ 下，
# 因此向上两级 parent.parent 即为项目根目录；用于把配置中的相对路径解析为绝对路径。
_ROOT = Path(__file__).resolve().parent.parent
# 全局配置文件的默认路径：<根>/config/config.yaml
CONFIG_PATH = _ROOT / "config" / "config.yaml"


class ConfigError(Exception):
    """配置加载/校验异常。配置文件缺失、格式错误或字段越界时抛出。"""


def _require(d: Dict, key: str, section: str) -> Any:
    """校验字典 d 中必须存在 key，缺失则抛出 ConfigError。

    section 仅用于错误信息中标注所属配置区块，便于定位问题。
    """
    if key not in d:
        raise ConfigError(f"配置缺失：[{section}] 缺少字段 '{key}'")
    return d[key]


def load_config(path: Path = CONFIG_PATH) -> Dict:
    """加载并校验配置。校验失败抛出 ConfigError。

    流程：读取 YAML → 校验必填字段 → 校验阈值范围 → 将相对路径转为绝对路径。
    返回校验并处理后的配置字典（其中路径字段已被就地改写为绝对路径）。
    """
    # 文件不存在直接报错，避免后续 open 抛出难以理解的异常
    path = Path(os.environ.get("AIOT_CONFIG_PATH", str(path)))
    if not Path(path).exists():
        raise ConfigError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 根节点必须是字典（映射），否则后续按 key 访问会失败
    if not isinstance(cfg, dict):
        raise ConfigError("配置文件格式错误：根节点应为字典")

    # 取出两个顶层区块：global（全局路径等）与 inference（推理参数）
    g = _require(cfg, "global", "root")
    inf = _require(cfg, "inference", "root")
    # 校验 global 区块的必填字段
    for k in ("capture_buffer", "buffer_101", "buffer_201", "logs_dir"):
        _require(g, k, "global")
    # 校验 inference 区块的必填字段
    for k in ("fps", "yolo_model", "yolo_conf", "person_min_pixels", "idle_timeout"):
        _require(inf, k, "inference")

    # 阈值越界校验：确保数值参数落在合理区间内
    if inf["fps"] <= 0:
        raise ConfigError("inference.fps 必须为正数")
    if float(inf.get("face_fps", inf["fps"])) <= 0 or float(inf.get("face_fps", inf["fps"])) > inf["fps"]:
        raise ConfigError("inference.face_fps 必须在 (0, inference.fps] 区间")
    if not (0.0 <= float(inf["yolo_conf"]) <= 1.0):
        raise ConfigError("inference.yolo_conf 必须在 [0,1] 区间")
    if inf["person_min_pixels"] < 0:
        raise ConfigError("inference.person_min_pixels 不能为负")
    if inf["idle_timeout"] <= 0:
        raise ConfigError("inference.idle_timeout 必须为正数")
    if not 0 <= float(inf.get("face_sim_threshold", 0.6)) <= 1:
        raise ConfigError("inference.face_sim_threshold 必须在 [0,1]")

    # 路径转为绝对（相对根目录解析）：仅对已配置且为相对路径的字段处理
    for k in ("capture_buffer", "buffer_101", "buffer_201", "annotated_buffer", "logs_dir"):
        if k in g and g[k] and not Path(g[k]).is_absolute():
            g[k] = str(_ROOT / g[k])
    # 模型路径相对根目录解析，避免依赖运行时 CWD
    # 仅当值包含路径分隔符时才当作路径解析（纯模型名如 "yolov8n.pt" 交由框架自行查找）
    ym = inf.get("yolo_model")
    if ym and ("/" in ym or "\\" in ym) and not Path(ym).is_absolute():
        inf["yolo_model"] = str(_ROOT / ym)
    # 人脸底库根目录：相对路径转绝对
    fr = inf.get("face_root")
    if fr and not Path(fr).is_absolute():
        inf["face_root"] = str(_ROOT / fr)
    return cfg
