# -*- coding: utf-8 -*-
"""配置层：加载并校验根目录 config/config.yaml（标注模块视角）。"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "config.yaml"


class ConfigError(Exception):
    """配置加载/校验异常。"""


def _require(d: Dict, key: str, section: str) -> Any:
    if key not in d:
        raise ConfigError(f"配置缺失：[{section}] 缺少字段 '{key}'")
    return d[key]


def load_config(path: Path = CONFIG_PATH) -> Dict:
    path = Path(os.environ.get("AIOT_CONFIG_PATH", str(path)))
    if not Path(path).exists():
        raise ConfigError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("配置文件格式错误：根节点应为字典")

    g = _require(cfg, "global", "root")
    tag = _require(cfg, "tag", "root")
    for k in ("buffer_101", "buffer_201", "annotated_buffer", "logs_dir"):
        _require(g, k, "global")
    seg = _require(tag, "segment_seconds", "tag")
    if seg <= 0:
        raise ConfigError("tag.segment_seconds 必须为正数")
    if float(tag.get("follow_poll_interval", 0.5)) <= 0:
        raise ConfigError("tag.follow_poll_interval 必须为正数")
    if len(tag.get("fourcc", "mp4v")) != 4:
        raise ConfigError("tag.fourcc 必须为 4 字符")

    for k in ("buffer_101", "buffer_201", "annotated_buffer", "logs_dir", "capture_buffer"):
        if k in g and g[k] and not Path(g[k]).is_absolute():
            g[k] = str(_ROOT / g[k])
    return cfg
