"""AIOT Video 桌面发布入口；也是冻结后各流水线子进程的统一入口。"""
from __future__ import annotations

import sys
import threading
import time
import webbrowser

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


def main() -> int:
    _configure_stdio()
    prepare_runtime_environment()
    if len(sys.argv) >= 3 and sys.argv[1] == "--internal-role":
        return _run_internal(sys.argv[2], sys.argv[3:])

    import uvicorn
    from webapp.server import create_app

    host, port = "127.0.0.1", 8080
    url = f"http://{host}:{port}"
    if "--no-browser" not in sys.argv[1:]:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    print(f"AIOT Video 已启动：{url}", flush=True)
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
