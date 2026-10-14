# -*- coding: utf-8 -*-
"""服务层：命令行入口（CLI 模式）。

本模块是 InsightfaceModel（AI 推理模块）的两种启动方式之一：
- CLI 模式（本文件）：前台运行单设备推理，适合调试和一次性批处理；
- API 模式（api.py）：Flask HTTP 服务（端口 5001），适合远程管控多设备。

用法:
    python InsightfaceModel/main.py --device-no device_0 --fps 10

--fps 是每秒目录内用于推理的帧数，与采集源 FPS 独立。

整体调用链：
    main() → load_config() → InferenceManager.start() → InferenceWorker 线程
    → ModelProcessor.run()（真正的推理主循环，见 model_processor.py）
    本文件只负责"参数解析 + 启动 + 等待结束"，不含任何推理业务逻辑。
"""
import argparse
import os
import sys
import time
from pathlib import Path
import signal

# 将项目根目录（1/1/）加入 sys.path，使 "InsightfaceModel.xxx" 这种
# 以包名为前缀的绝对导入在任何工作目录下都能生效
# （os.path.dirname 两次：main.py → InsightfaceModel/ → 项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InsightfaceModel.config_loader import load_config, ConfigError  # 配置加载 + 配置异常类型
from InsightfaceModel.manager import InferenceManager                # 多设备推理线程管理器
from InsightfaceModel.logger import logger                           # 统一日志（logs/<日期>/inference.log）


def main():
    """解析命令行参数，启动一路推理任务并等待其结束。"""
    # ---- 命令行参数定义 ----
    parser = argparse.ArgumentParser(description="InsightfaceModel 推理任务") # 程序描述，执行 -h 时会展示，创建解析器对象
    parser.add_argument("--device-no", default="device_0", help="设备编号，如 device_0")#add_argument() 方法用于向解析器添加参数，指定参数名称、默认值和帮助信息
    # --fps 为 None 时由 ModelProcessor 回退到 config.yaml 中 inference.fps 的值
    parser.add_argument("--fps", type=int, default=None, help="抽帧帧率，默认读配置")
    args = parser.parse_args()#args为一个命名空间对象，包含了所有解析后的参数值，可以通过 args.device_no 和 args.fps 访问对应的参数值

    # ---- 加载全局配置（config/config.yaml）----
    # 配置有误（缺字段/路径错/阈值越界）属于致命问题，直接退出，不做任何降级
    try:
        config = load_config()
    except ConfigError as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    # ---- 启动推理 ----
    manager = InferenceManager(config) # 3. 初始化推理管理器实例，传入全局配置
    # start() 创建 InferenceWorker 守护线程并启动，立即返回设备号（int）；
    # 真正的推理（扫描 capture_buffer → 检测 → 编码落盘）在后台线程中进行。
    # 若该设备已有任务在运行，这里会抛 RuntimeError（单设备单任务）。
    dno = manager.start(args.device_no, fps=args.fps)
    worker = manager._workers[dno]
    def request_stop(*_):
        worker.stop_event.set()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    state = Path(os.environ["AIOT_PIPELINE_STATE"]) if os.environ.get("AIOT_PIPELINE_STATE") else None
    result = 0
    logger.info("推理任务已启动：设备 %d，按 Ctrl+C 停止", dno)

    # ---- 等待任务结束 ----
    # 推理线程的自然结束方式有两种：连续 idle_timeout 秒无新帧自动回收（批处理
    # 场景"跑完即走"），或线程内部异常退出；两种情况下 alive 都会变 False，
    # 主线程随之退出等待循环。
    try:
        # status() 返回列表；先判非空再取 [0]，防止极端情况下设备刚被移除导致越界
        while manager.status(dno) and manager.status(dno)[0]["alive"]:
            if state and (state / "inference.stop").exists():
                request_stop()
            time.sleep(0.2)
        logger.info("设备 %d 任务已结束（空闲回收或异常）", dno)
        if worker.error:
            logger.error("推理任务失败: %s", worker.error)
            result = 1
    except KeyboardInterrupt:
        # 用户按 Ctrl+C：不视为错误，继续走 finally 做收尾
        logger.info("收到中断信号，正在停止...")
    finally:
        # 无论正常结束还是中断，都停止全部推理线程：
        # 置 stop_event → 等线程退出 → ModelProcessor 释放模型/GPU 资源。
        # 注意即使任务已自然结束也要调用，stop() 对已退出的线程是安全的。
        manager.stop()
    return result


# 标准入口保护：仅当本文件被直接执行（而非被 import）时才运行 main()
if __name__ == "__main__":
    raise SystemExit(main())
