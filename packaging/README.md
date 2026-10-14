# Windows 发布说明

当前采用 PyInstaller `onedir` 方案。程序资源保存在发布目录，配置、日志、上传、缓存和输出写入：

```text
%LOCALAPPDATA%\AIOTVideo
```

## 构建

在项目虚拟环境安装构建依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\packaging\build_windows.ps1
```

构建结果位于 `dist\AIOTVideo`。第一阶段保留控制台窗口，便于内测期间查看启动错误。发布前需要在一台没有安装 Python 的 Windows 电脑上验证完整视频处理流程。

## 入口设计

`app_launcher.py` 是发布入口。冻结后的同一个 EXE 通过隐藏参数分派网站、总流水线、采集、推理和标注进程，避免依赖目标电脑上的 Python。
