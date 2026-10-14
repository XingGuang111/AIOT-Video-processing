# 本文件负责识别和整理输入源，不负责真正打开视频。
# OpenCV 打开视频的逻辑位于 capture_worker.py。
from typing import Tuple, Optional
from urllib.parse import urlparse
from .logger import logger

# 常见视频文件扩展名，用于区分“远程视频文件”和“在线直播流”。
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".ts"}


def _is_video_url(url: str) -> bool:
    """判断某个 HTTP/HTTPS 地址是否指向视频文件，而不是直播流。"""
    # 只检查 URL 的路径部分，避免查询参数影响扩展名判断。
    path = urlparse(url).path.lower()
    # 路径以任意一个视频扩展名结尾，就认为它指向视频文件。
    return any(path.endswith(ext) for ext in VIDEO_EXTS)


def is_ipv4(address: str) -> bool:
    """判断字符串是否为 IPv4 地址，用于纯 IP 摄像头场景。"""
    try:
        # IPv4 必须由四段数字组成，例如 192.168.1.10。
        parts = address.strip().split(".")
        if len(parts) != 4:
            return False
        # 每一段都必须能转成整数，且范围在 0 到 255 之间。
        return all(0 <= int(part) <= 255 for part in parts)
    except (ValueError, AttributeError):
        # 输入不是字符串或包含非数字内容时，判定为非法 IPv4。
        return False


def parse_source(source: str, account: Optional[str] = None, password: Optional[str] = None) -> Tuple[str, str]:
    """按输入源类型返回 (source_type, final_source)。

    这是 CaptureManager.start 的关键前置步骤：
    - source = '0' -> device
    - source = 'rtsp://...' -> rtsp
    - source = '192.168.1.10' -> 需要 account/password，构造成 RTSP URL
    - source = 'http://.../movie.mp4' -> file
    - source = 'http://.../stream' -> stream
    """
    # 清除输入两端空格，避免 " 0 " 或 " video.mp4 " 判断失败。
    source = source.strip()

    # 纯数字表示本机摄像头编号，例如 "0" 表示第一个摄像头。
    if source.isdigit():
        return "device", source

    # 已经是完整 RTSP 地址时直接使用，不再重复拼接地址。开头匹配忽略大小写
    if source.lower().startswith("rtsp://"):
        return "rtsp", source

    # 纯 IPv4 地址表示用户只提供了摄像头 IP，需要账号和密码来拼接 RTSP 地址。
    if is_ipv4(source):
        if not account or not password:
            raise ValueError("纯IP摄像头需要传入account和password参数")
        rtsp_url = f"rtsp://{account}:{password}@{source}:554/Streaming/Channels/101"
        return "rtsp", rtsp_url

    # HTTP/HTTPS 地址还要继续区分：视频文件 URL，或在线直播流 URL。
    if source.lower().startswith(("http://", "https://")):
        if _is_video_url(source):
            return "file", source
        return "stream", source

    # 其他输入默认按本地视频文件路径处理。
    return "file", source


def desensitize_rtsp_url(url: str) -> str:
    """对 RTSP URL 中的认证信息做脱敏处理，避免日志泄露密码。

    调用链：
    CaptureWorker.__init__ 中 self.status = CaptureStatus(..., source=desensitize_rtsp_url(source))
    """
    # 非 RTSP 地址不按认证信息处理，直接原样返回。
    if not url.lower().startswith("rtsp://"):
        return url
    try:
        # 取出 rtsp:// 后的部分，并检查是否存在 user:password@host 结构。
        prefix = url.split("://")[1]
        if "@" in prefix:
            auth_part = prefix.split("@")[0]
            if ":" in auth_part:
                # 只隐藏密码，保留用户名和地址，便于日志识别设备。
                user, pwd = auth_part.split(":", 1)
                return url.replace(f":{pwd}@", ":***@")
    except Exception as e:
        # 脱敏只影响日志，不应阻断视频采集，因此记录警告后继续。
        logger.warning(f"RTSP脱敏失败: {e}")
    # 没有认证信息或格式不符合预期时，返回原地址。
    return url
