import os
from typing import Tuple, Optional
from urllib.parse import urlparse
from .logger import logger

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".ts"}


def _is_video_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in VIDEO_EXTS)


def is_ipv4(address: str) -> bool:
    try:
        parts = address.strip().split(".")
        if len(parts) != 4:
            return False
        return all(0 <= int(part) <= 255 for part in parts)
    except (ValueError, AttributeError):
        return False


def parse_source(source: str, account: Optional[str] = None, password: Optional[str] = None) -> Tuple[str, str]:
    source = source.strip()
    if source.isdigit():
        return "device", source
    if source.lower().startswith("rtsp://"):
        return "rtsp", source
    if is_ipv4(source):
        if not account or not password:
            raise ValueError("纯IP摄像头需要传入account和password参数")
        rtsp_url = f"rtsp://{account}:{password}@{source}:554/Streaming/Channels/101"
        return "rtsp", rtsp_url
    if source.lower().startswith(("http://", "https://")):
        if _is_video_url(source):
            return "file", source
        return "stream", source
    return "file", source


def desensitize_rtsp_url(url: str) -> str:
    if not url.lower().startswith("rtsp://"):
        return url
    try:
        prefix = url.split("://")[1]
        if "@" in prefix:
            auth_part = prefix.split("@")[0]
            if ":" in auth_part:
                user, pwd = auth_part.split(":", 1)
                return url.replace(f":{pwd}@", ":***@")
    except Exception as e:
        logger.warning(f"RTSP脱敏失败: {e}")
    return url
