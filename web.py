"""本地网站启动入口：.venv/Scripts/python.exe web.py"""
import argparse
import uvicorn
from webapp.server import create_app


def main():
    parser = argparse.ArgumentParser(description="AIOT 视频流水线网站")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app = create_app()
    print(f"网站已启动：http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
