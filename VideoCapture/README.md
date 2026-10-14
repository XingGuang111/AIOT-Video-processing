# VideoCapture

全帧采集模块：读取本地文件、RTSP、HTTP 视频源或摄像头，按设备和秒级时间戳原子发布 JPEG 与 `_ready.json`。

推荐通过项目根目录的 `web.py` 或 `pipeline.py` 启动完整流水线。安装、缓存模式、输出格式和测试方法统一见 [项目 README](../README.md)。

独立采集：

```powershell
.\.venv\Scripts\python.exe VideoCapture/main.py extract "D:\video.mp4" --device-no 0
```

独立采集 API：

```powershell
.\.venv\Scripts\python.exe VideoCapture/main.py server
```

采集配置为本目录 `config.yaml`。完整流水线会覆盖输出路径、起始时间和积压限制；JPEG 质量与重连设置仍来自采集配置。

`videocapture/buffer_cleaner.py` 仅用于独立采集 API 的容量清理，完整流水线使用根目录的 `buffer_cleanup.py`，按下游消费进度清理。两者不应同时操作同一设备目录。
