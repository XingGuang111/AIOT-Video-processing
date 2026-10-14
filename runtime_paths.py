"""开发环境与 PyInstaller 发布环境共用的路径、子进程命令。"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


APP_NAME = "AIOTVideo"
SOURCE_ROOT = Path(__file__).resolve().parent
FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT)).resolve()


def _default_data_root() -> Path:
    if not FROZEN:
        return SOURCE_ROOT
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / APP_NAME if local else Path.home() / f".{APP_NAME}"


DATA_ROOT = Path(os.environ.get("AIOT_DATA_ROOT", _default_data_root())).resolve()


def prepare_runtime_environment() -> Path:
    """创建可写数据目录，并返回本次使用的配置文件。"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AIOT_RESOURCE_ROOT", str(RESOURCE_ROOT))
    os.environ.setdefault("AIOT_DATA_ROOT", str(DATA_ROOT))
    cache_root = DATA_ROOT / "cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    os.environ["YOLO_CONFIG_DIR"] = str(cache_root / "ultralytics")

    bundled = RESOURCE_ROOT / "config" / "config.yaml"
    config_dir = DATA_ROOT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if FROZEN and not config_path.exists():
        shutil.copy2(bundled, config_path)
    elif not FROZEN:
        config_path = bundled
    os.environ.setdefault("AIOT_CONFIG_PATH", str(config_path))
    return config_path


_ROLE_SCRIPTS = {
    "pipeline": "pipeline.py",
    "capture": "VideoCapture/main.py",
    "inference": "InsightfaceModel/main.py",
    "tag": "VideoTag/main.py",
}


def role_command(role: str, *arguments: str) -> list[str]:
    """生成开发环境或冻结 EXE 中等价的内部进程命令。"""
    if role not in _ROLE_SCRIPTS:
        raise ValueError(f"未知内部角色：{role}")
    if FROZEN:
        return [sys.executable, "--internal-role", role, *map(str, arguments)]
    return [sys.executable, "-u", str(RESOURCE_ROOT / _ROLE_SCRIPTS[role]), *map(str, arguments)]
