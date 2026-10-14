# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH).parent
datas = [
    (str(root / "webapp" / "static"), "webapp/static"),
    (str(root / "config" / "config.yaml"), "config"),
    (str(root / "VideoCapture" / "config.yaml"), "VideoCapture"),
]

required_models = [
    root / "config" / "models" / "yolov8s.pt",
    root / "config" / "models" / "buffalo_l" / "det_10g.onnx",
    root / "config" / "models" / "buffalo_l" / "w600k_r50.onnx",
]
for model in required_models:
    if not model.is_file():
        raise SystemExit(f"缺少发布模型：{model}")
    datas.append((str(model), str(model.parent.relative_to(root))))

a = Analysis(
    [str(root / "app_launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "ultralytics.trackers.byte_tracker",
        "ultralytics.trackers.bot_sort",
        "ultralytics.trackers.track",
    ],
    hookspath=[],
    excludes=[
        "pytest", "IPython", "jupyter", "matplotlib.tests", "tkinter",
        "polars", "onnxruntime.quantization", "onnxruntime.tools",
        "onnxruntime.transformers", "insightface.commands", "insightface.gui",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIOTVideo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    version=str(root / "packaging" / "version_info.txt"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AIOTVideo",
)
