# -*- coding: utf-8 -*-
"""视频帧提取工具 —— 命令行入口。

用法:
    python main.py server                                            # 启动 API 服务
    python main.py extract SOURCE [--sampling -1]                    # 命令行直接提取（单路）
    python main.py extract SOURCE [--sampling 1]                     # 每秒抽 1 帧（降采样）
"""
import argparse
import sys
import yaml
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def cmd_server():
    from app import app
    import uvicorn
    from videocapture import logger
    logger.info("🚀 视频帧采集服务启动，监听端口 8000")
    logger.info("📖 接口文档地址：http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_extract(source: str, sampling: float):
    from videocapture import CaptureManager, logger
    mgr = CaptureManager(config)
    device_no = mgr.start(source=source, sampling=sampling)
    mode = "全量抽帧（每帧都写）" if sampling == -1 else f"每 {sampling} 秒抽 1 帧"
    logger.info("设备 %d 已启动（%s），按 Ctrl+C 停止", device_no, mode)
    terminal_states = {"finished", "stopped", "error"}
    try:
        while True:
            status_list = mgr.status(device_no)
            if status_list:
                s = status_list[0]
                logger.info(
                    "设备 %d | 状态: %s | 已采集: %d 帧 | 分辨率: %s",
                    device_no, s["status"], s["frames_captured"], s["resolution"]
                )
                if s["status"] in terminal_states:
                    logger.info("设备 %d 采集结束，状态: %s", device_no, s["status"])
                    break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        mgr.stop(device_no)
        logger.info("已停止")


def main():
    parser = argparse.ArgumentParser(
        description="视频帧采集 / 提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("server", help="启动 FastAPI 服务（端口 8000）")

    p_extract = sub.add_parser("extract", help="命令行直接提取单路视频（前台运行）")
    p_extract.add_argument("source", help="视频源：本地文件 / RTSP URL / 摄像头号（0,1,2...）")
    p_extract.add_argument(
        "--sampling", type=float, default=-1,
        help="采样间隔（秒）。-1 = 全量抽帧（每帧都写，推荐原帧率处理），默认 -1；"
             "其他正数 = 每 N 秒抽 1 帧（降采样）。",
    )

    args = parser.parse_args()
    if args.cmd == "server":
        cmd_server()
    elif args.cmd == "extract":
        cmd_extract(args.source, args.sampling)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
