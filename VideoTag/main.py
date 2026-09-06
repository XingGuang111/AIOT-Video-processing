# -*- coding: utf-8 -*-
"""服务层：命令行入口。

    python VideoTag/main.py --source ./test.mp4 --device-no device_0
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VideoTag.config_loader import load_config, ConfigError
from VideoTag.manager import TagManager
from VideoTag.logger import logger


def main():
    parser = argparse.ArgumentParser(description="VideoTag 视频标注任务")
    parser.add_argument("--source", required=True, help="视频路径（本地文件）/ NVR 地址")
    parser.add_argument("--device-no", default="device_0", help="设备编号，如 device_0")
    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    manager = TagManager(config)
    dno = manager.start(args.source, args.device_no)
    logger.info("标注任务已启动：设备 %d，按 Ctrl+C 停止", dno)
    try:
        while manager.status(dno) and manager.status(dno)[0]["alive"]:
            time.sleep(2)
        st = manager.status(dno)[0]
        if st["error"]:
            logger.error("设备 %d 标注失败: %s", dno, st["error"])
        else:
            logger.info("设备 %d 标注完成，共 %d 段", dno, st["segments"])
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        manager.stop()


if __name__ == "__main__":
    main()
