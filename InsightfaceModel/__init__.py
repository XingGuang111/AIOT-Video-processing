# -*- coding: utf-8 -*-
"""InsightfaceModel —— AI 推理模块：人体检测 + 人脸 512 维特征 + 跟踪 + 101/201 协议。"""
from .logger import logger
from .config_loader import load_config, ConfigError
from .manager import InferenceManager, parse_device_no

__all__ = ["logger", "load_config", "ConfigError", "InferenceManager",
           "parse_device_no", "start_inference"]


def start_inference(device_no="device_0", fps=None, config=None):
    """便捷函数：启动单设备推理并返回 (manager, device_no)。"""
    cfg = config or load_config()
    manager = InferenceManager(cfg)
    dno = manager.start(device_no, fps=fps)
    return manager, dno
