# -*- coding: utf-8 -*-
"""API 层：Flask HTTP 服务（端口 5001），启停推理任务。仅调用服务层。

    python InsightfaceModel/api.py --host 0.0.0.0 --port 5001
"""
import argparse
import os
import sys

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InsightfaceModel.config_loader import load_config
from InsightfaceModel.manager import InferenceManager
from InsightfaceModel.logger import logger

app = Flask(__name__)
config = load_config()
manager = InferenceManager(config)


def _ok(msg="ok", data=None):
    return jsonify({"code": 0, "msg": msg, "data": data})


def _err(msg, code=500):
    return jsonify({"code": code, "msg": msg, "data": None}), code


@app.post("/api/v1/inference/start")
def start():
    body = request.get_json(silent=True) or {}
    device_no = body.get("device_no", "device_0")
    fps = body.get("fps")
    try:
        dno = manager.start(device_no, fps=fps)
        return _ok(f"设备 {dno} 推理任务已启动", {"device_no": dno})
    except Exception as e:
        logger.exception("启动推理失败")
        return _err(str(e))


@app.post("/api/v1/inference/stop")
def stop():
    body = request.get_json(silent=True) or {}
    device_no = body.get("device_no")
    try:
        stopped = manager.stop(device_no)
        return _ok(f"已停止 {len(stopped)} 个设备", {"stopped": stopped})
    except KeyError as e:
        return _err(str(e), code=404)
    except Exception as e:
        logger.exception("停止推理失败")
        return _err(str(e))


@app.get("/api/v1/inference/status")
def status():
    device_no = request.args.get("device_no")
    try:
        return _ok("ok", manager.status(device_no))
    except KeyError as e:
        return _err(str(e), code=404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightfaceModel API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    logger.info("InsightfaceModel API 启动: http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)
