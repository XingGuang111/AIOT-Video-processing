# -*- coding: utf-8 -*-
"""统一日志工具：日志写入根目录 logs/<日期>/inference.log，并输出到控制台。"""
import logging
from pathlib import Path
from datetime import datetime

# 根目录 = InsightfaceModel 的上一级
_ROOT = Path(__file__).resolve().parent.parent


def init_logger(name: str = "inference") -> logging.Logger:
    """初始化分模块日志，按日期目录轮转。"""
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = _ROOT / "logs" / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(f"insightface_model.{name}")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
    return logger


logger = init_logger()
