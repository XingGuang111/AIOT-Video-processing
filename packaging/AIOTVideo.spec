# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


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

binaries = []
hiddenimports = collect_submodules("InsightfaceModel")
hiddenimports += collect_submodules("VideoCapture")
hiddenimports += collect_submodules("VideoTag")
for package in ("ultralytics", "insightface", "onnxruntime", "imageio_ffmpeg"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [str(root / "app_launcher.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["pytest", "IPython", "jupyter", "matplotlib.tests"],
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AIOTVideo",
)
