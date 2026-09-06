# -*- coding: utf-8 -*-
"""VideoTag —— 视频标注生成模块：协议解码 + 时间戳关联 + 分段标注 + H.264 输出。"""
from .logger import logger
from .config_loader import load_config, ConfigError
from .manager import TagManager, parse_device_no

__all__ = ["logger", "load_config", "ConfigError", "TagManager", "parse_device_no"]
