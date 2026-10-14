# -*- coding: utf-8 -*-
"""服务层：VideoTag 命令行入口。

用法:
    python VideoTag/main.py --source ./test.mp4 --device-no device_0

职责：解析命令行参数 → 加载配置 → 启动标注任务 → 轮询等待完成/处理中断。
真正的标注逻辑在 TagManager → TagWorker → SegmentProcessor 中执行，
本文件只是最外层的"命令行壳"（不碰任何视频/标注细节）。
"""
import argparse
import os
import sys
import time
from pathlib import Path
import signal

# 将项目根目录加入 sys.path，使 `import VideoTag.xxx` 生效
# （无论从哪个目录启动本脚本，都能找到同级的 VideoTag 包）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VideoTag.config_loader import load_config, ConfigError
from VideoTag.manager import TagManager
from VideoTag.logger import logger


def main():
    """程序入口：解析参数 → 加载配置 → 启动并等待标注任务结束。"""
    parser = argparse.ArgumentParser(description="VideoTag 视频标注任务")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", "--video-path", dest="source", help="批处理视频路径（保留旧 --source）")
    mode.add_argument("--follow", action="store_true", help="轮询帧目录与配对协议，逐秒输出")
    parser.add_argument("--device-no", default="device_0", help="设备编号，如 device_0")
    args = parser.parse_args()

    # 1) 加载 config/config.yaml（失败则打日志并以非零码退出）
    try:
        config = load_config()
    except ConfigError as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    # 2) 创建标注管理器，启动该设备的标注线程（start 返回归一化后的设备号）
    manager = TagManager(config)#初始化多设备标注管理器
    dno = manager.start(args.source or "", args.device_no, follow=args.follow)
    worker = manager._workers[dno]
    def request_stop(*_):
        worker.stop_event.set()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    state = Path(os.environ["AIOT_PIPELINE_STATE"]) if os.environ.get("AIOT_PIPELINE_STATE") else None
    result = 0
    logger.info("标注任务已启动：设备 %d，按 Ctrl+C 停止", dno)
    try:
        # 3) 轮询等待：只要线程还活着（alive=True）就每 2 秒查一次状态
        while manager.status(dno) and manager.status(dno)[0]["alive"]:
            if state and (state / "tag.stop").exists():
                request_stop()
            time.sleep(0.2)
        # 4) 线程结束（跑完/出错/被停止）后，取最终状态报告结果
        st = manager.status(dno)[0]
        if st["error"]:
            logger.error("设备 %d 标注失败: %s", dno, st["error"])
            result = 1
        else:
            logger.info("设备 %d 标注完成，共 %d 段", dno, st["segments"])
    except KeyboardInterrupt:
        # Ctrl+C：请求停止标注任务（置位停止信号，线程自然退出）
        logger.info("收到中断信号，正在停止...")
        manager.stop()
    finally:
        manager.stop()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
