"""AIOT Video 桌面发布入口；也是冻结后各流水线子进程的统一入口。"""
from __future__ import annotations

import sys
import json
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from app_version import APP_NAME, APP_VERSION
from runtime_paths import prepare_runtime_environment


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _run_internal(role: str, arguments: list[str]) -> int:
    sys.argv = [role, *arguments]
    if role == "pipeline":
        from pipeline import main
    elif role == "capture":
        from VideoCapture.main import main
    elif role == "inference":
        from InsightfaceModel.main import main
    elif role == "tag":
        from VideoTag.main import main
    else:
        raise SystemExit(f"未知内部角色：{role}")
    result = main()
    return int(result or 0)


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url)


def _port_state(url: str, host: str, port: int) -> str:
    """返回 free、aiot 或 occupied。"""
    try:
        with urlopen(f"{url}/api/overview", timeout=1.0) as response:
            body = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and {"jobs", "devices"}.issubset(body):
                return "aiot"
    except (OSError, ValueError, URLError):
        pass
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return "occupied"
    except OSError:
        return "free"


def _show_error(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, f"{APP_NAME} 启动失败", 0x10)
        except OSError:
            pass


def main() -> int:
    _configure_stdio()
    prepare_runtime_environment()
    if len(sys.argv) >= 3 and sys.argv[1] == "--internal-role":
        return _run_internal(sys.argv[2], sys.argv[3:])

    import uvicorn
    from webapp.server import create_app

    host, port = "127.0.0.1", 8080
    url = f"http://{host}:{port}"
    port_state = _port_state(url, host, port)
    if port_state == "aiot":
        webbrowser.open(url)
        print(f"{APP_NAME} 已经在运行，已打开现有页面。", flush=True)
        return 0
    if port_state == "occupied":
        _show_error(f"端口 {port} 已被其他程序占用，请关闭占用程序后重试。")
        return 2
    if "--no-browser" not in sys.argv[1:]:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    print(f"{APP_NAME} {APP_VERSION} 已启动：{url}", flush=True)
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
