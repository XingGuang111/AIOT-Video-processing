# -*- coding: utf-8 -*-
"""服务层：命令行入口。

    python InsightfaceModel/main.py --device-no device_0 --fps 10
"""
import argparse
import os
import sys
import time

# 将项目根目录加入 sys.path，使包内相对导入生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InsightfaceModel.config_loader import load_config, ConfigError
from InsightfaceModel.manager import InferenceManager
from InsightfaceModel.logger import logger


def main():
    parser = argparse.ArgumentParser(description="InsightfaceModel 推理任务") # 程序描述，执行 -h 时会展示，创建解析器对象
    parser.add_argument("--device-no", default="device_0", help="设备编号，如 device_0")
    parser.add_argument("--fps", type=int, default=None, help="抽帧帧率，默认读配置")
    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    manager = InferenceManager(config) # 3. 初始化推理管理器实例，传入全局配置
    dno = manager.start(args.device_no, fps=args.fps)
    logger.info("推理任务已启动：设备 %d，按 Ctrl+C 停止", dno)
    try:
        while manager.status(dno) and manager.status(dno)[0]["alive"]:
            time.sleep(5)
        logger.info("设备 %d 任务已结束（空闲回收或异常）", dno)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
