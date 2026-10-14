# AIOT 视频分析工作台

对本地视频、RTSP 和摄像头画面进行人体检测、跟踪及人脸检测，输出绿色人体框、红色人脸框的标注视频。网站使用 FastAPI；采集、推理和标注由三个独立进程并行处理。

## 安装与启动

在项目根目录运行：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web.py
```

已有环境只需执行最后一条，或双击 `start_website.cmd`。打开 `http://127.0.0.1:8080/`。局域网使用 `web.py --host 0.0.0.0 --port 8080`；当前网站没有账号认证，适用于可信本机或内网。

GPU 推理需要与本机 CUDA 环境匹配的 PyTorch 和 ONNX Runtime GPU。普通 `onnxruntime` 是 CPU 后端，不要与 `onnxruntime-gpu` 同时安装。`global.gpu: true` 表示优先使用 GPU，实际执行设备以网站任务详情和初始化日志为准。

模型文件为 `config/models/yolov8s.pt`、`config/models/buffalo_l/det_10g.onnx` 和 `config/models/buffalo_l/w600k_r50.onnx`。缺少权重时加载器尝试下载，离线使用应提前准备。模型及示例视频不随 Git 仓库提交，可使用自己的视频路径。

## 网站操作

1. 上传视频（最大 2 GB）、填写本机视频路径或 RTSP 地址，选择设备编号及缓存模式，然后开始分析。同一设备同时只能运行一个任务。
2. “03 标注画面”支持顺序播放、暂停、继续和拖拽回看。进度条只开放已生成的部分；初次播放先缓冲两段，后续结果未就绪时等待，不跳到最新秒。暂停与继续复用连接，拖拽直接定位目标片段。
3. “04 输出片段”只展示当前选中任务；切换或新建任务立即清空旧列表。可回看片段、下载一秒视频和完整视频。
4. 任务详情显示实际推理设备和各步骤累计、平均耗时；日志展示采集、推理、标注、缓存清理和完整视频合并过程。

刷新或重新打开页面时不显示上次任务、视频和日志；勾选“查看之前的任务”后手动选择。刷新不影响后台处理，关闭网站服务会停止它启动的任务。

“清理记录”删除已结束任务的记录及对应任务日志，保留源视频、标注视频和活动任务。上传视频在 `logs/web_uploads/`，不会自动删除。

## 缓存模式

| 模式 | 原始帧、101/201 协议、逐帧标注 JSON | 所选设备的历史中间缓存 |
| --- | --- | --- |
| 运行模式 | 标注秒段保存成功后立即删除；占用文件每 250 毫秒重试 | 清理 |
| 测试模式（网站默认） | 全部保留，方便检查各步骤结果 | 保留 |

两种模式均保留完整标注视频、一秒片段和日志。运行模式按下游消费进度动态清理，不等整段处理完成；跳过且不再使用的缓存也会清理。其他设备不受影响。

`global.max_buffer_seconds` 默认 8，采集积压达到上限时暂停。运行模式将已消费但暂未删除的原帧目录也计入上限。测试模式只限制待消费积压，已处理文件持续保留。上限按秒目录计算，并非固定 MB 或内存上限；成片和日志仍随运行时间增长。

## 处理流程与编号

```text
VideoCapture：全量 JPEG + 完成标记
  → InsightfaceModel：YOLO + ByteTrack、InsightFace
  → 101/201 协议 + 抽样帧标注 JSON
  → VideoTag：逐帧插值绘框、一秒视频
  → FFmpeg：无损拼接完整标注视频
```

默认每秒抽样 10 帧进行人体和人脸推理，采集及输出保留源帧率，不因推理抽样丢弃其他视频帧。绿框和红框按对应目标插值，无法匹配的身份不会互相插值。

人体 `ID` 跨帧保留，三秒内位置相符的 ByteTrack 断轨尝试一对一恢复；严重遮挡或交叉仍可能误判。人脸显示编号每帧从 `F:0` 开始，内部关联独立于显示编号。`Total` 是已分配人体轨迹编号的最大值，不等于准确去重人数。

本地文件默认尽快处理；实时流可能因推理速度和缓存限流产生延迟。预览通过有界队列和 MJPEG 按原帧率顺序输出。完整视频使用 `imageio-ffmpeg` 附带的 FFmpeg 拼接，不再次编码。

## 输出目录

```text
capture_buffer/device_N/时间戳/
  frame_000000.jpg …
  _ready.json
101_buffer/device_N/
  时间戳.bin                  # 秒内聚合人体记录
  时间戳.frames.json          # 抽样帧人体、人脸位置
201_buffer/device_N/
  时间戳.bin                  # 秒内聚合人脸记录、512 维特征
annotated_buffer/device_N/
  segments/时间戳.mp4         # 一秒片段，尾段保持实际时长
  起始时间戳_结束时间戳_full.mp4
logs/
  pipeline_设备号_运行标识/   # 任务配置、进程状态、计时、汇总
  web_jobs/                  # 网站任务记录与日志
  web_uploads/               # 上传源视频
```

有缺失或失败时，合并视频使用 `_partial.mp4`，需查看日志；中途停止保留已发布片段。

## 命令行与配置

```powershell
.\.venv\Scripts\python.exe pipeline.py --source "D:\video.mp4" --device-no 0 --buffer-mode runtime
.\.venv\Scripts\python.exe pipeline.py --source "D:\video.mp4" --device-no 0 --buffer-mode test
```

本地视频处理完自动结束，实时源按 Ctrl+C 停止。省略 `--buffer-mode` 遵循配置清理开关，不启用运行模式的历史缓存清理。

| 配置 | 用途 |
| --- | --- |
| `config/config.yaml` | 全局路径、GPU、抽样率、检测阈值、标注样式和编码 |
| `VideoCapture/config.yaml` | JPEG 质量、采集默认帧率和重连 |
| `AIOT_CONFIG_PATH` | 覆盖全局配置路径，供启动器、推理和标注读取 |

网站为每个任务生成独立配置，不修改全局文件。`tag.segment_seconds` 只用于独立批处理标注，网站流水线始终一秒发布。三个进程并行，总处理项包含子步骤，各项耗时不能直接相加。

独立调试入口仍保留：

```powershell
.\.venv\Scripts\python.exe VideoCapture/main.py extract "D:\video.mp4" --device-no 0
.\.venv\Scripts\python.exe InsightfaceModel/main.py --device-no 0
.\.venv\Scripts\python.exe VideoTag/main.py --follow --device-no 0
```

独立 API 在 `VideoCapture/app.py`、`InsightfaceModel/api.py` 和 `VideoTag/api.py`，不参与网站请求处理。独立采集 API 的容量清理器与流水线消费清理不同，避免同时操作同一设备目录。

## 项目文件结构

下面的 `1/` 指当前项目根目录 `D:\笔记\python程序\1\1`。源码、配置与运行产物分开存放：

```text
1/
├── README.md                        # 使用说明、架构、目录和验证方法
├── requirements.txt                 # 网站及三个处理模块的 Python 依赖
├── start_website.cmd                # Windows 启动脚本
├── web.py                           # 网站进程入口
├── pipeline.py                      # 单设备三进程流水线入口
├── buffer_cleanup.py                # 运行模式动态缓存清理
├── capture_buffers.py               # 缓存目录扫描与容量统计
├── runtime_metrics.py               # 各阶段耗时、状态原子写盘
├── video_output.py                  # FFmpeg 拼接完整视频
├── .gitignore                       # 排除模型、视频、日志等运行文件
├── ATIO视频处理.drawio              # 可用 draw.io 打开的设计图
├── webapp/
│   ├── __init__.py                  # 网站 Python 包标记
│   ├── server.py                    # FastAPI 路由与应用生命周期
│   ├── service.py                   # 任务管理、结果和日志读取
│   ├── preview.py                   # 后台解码与有界帧队列
│   └── static/
│       ├── index.html               # 网站页面结构
│       ├── style.css                # 页面样式和移动端布局
│       ├── app.js                   # 表单、任务选择、轮询和数据显示
│       └── player.js                # 播放、暂停、拖拽与逐帧进度
├── VideoCapture/
│   ├── main.py                     # 独立采集 / API 启动入口
│   ├── app.py                      # 独立采集 FastAPI
│   ├── config.yaml                 # JPEG、默认 FPS、重连等配置
│   ├── README.md、requirements.txt  # 独立采集说明与依赖
│   ├── videocapture/               # 采集线程、管理器、源解析等
│   └── demo_videos/                # 本地示例视频，不随仓库提交
├── InsightfaceModel/
│   ├── main.py、api.py              # 推理 CLI 与独立 HTTP API
│   ├── manager.py                  # 推理任务线程管理
│   ├── model_processor.py          # 按秒抽样、推理与结果发布
│   ├── inference_engine.py         # YOLO、InsightFace 模型封装
│   ├── tracker.py                  # 人体编号与短暂断轨恢复
│   ├── protocol_encoder.py         # 101/201 二进制协议编码
│   └── config_loader.py、logger.py、__init__.py
├── VideoTag/
│   ├── main.py、api.py              # 标注 CLI 与独立 HTTP API
│   ├── manager.py                  # 标注任务线程管理
│   ├── segment_processor.py        # 逐帧绘框与视频分段发布
│   ├── frame_annotations.py        # 抽样框到输出帧的插值
│   ├── protocol_decoder.py         # 101/201 协议读取与校验
│   └── config_loader.py、logger.py、__init__.py
├── config/
│   ├── config.yaml                 # 流水线全局配置
│   └── models/                     # YOLO 与 InsightFace 模型
├── tools/                          # 回归测试和浏览器验收脚本
├── capture_buffer/                 # 原始帧，详见“输出目录”
├── 101_buffer/、201_buffer/         # 推理协议与逐帧标注
├── annotated_buffer/               # 标注视频与一秒片段
├── logs/                           # 任务日志、上传文件、验证产物
├── assets/                         # 保留的旧说明配图，运行不依赖
├── .venv/                          # 本机 Python 虚拟环境
└── __pycache__/                    # Python 自动生成的字节码缓存
```

`.git/` 是版本管理数据，`.claude/` 是本地开发工具目录，都不参与视频处理。运行产生的目录可能为空，模型、示例视频和虚拟环境需要按本机环境准备。

## webapp 的作用与调用关系

`webapp/` 是网站控制与展示层：接收浏览器操作、启动或停止流水线、读取本地结果和日志，并将标注视频提供给播放器。真正的模型推理在 `InsightfaceModel` 子进程内执行，HTTP 请求不直接加载模型。

| 文件 | 主要内容 | 与其他文件的关系 |
| --- | --- | --- |
| `webapp/server.py` | `create_app()` 创建 FastAPI；定义页面、上传、任务启停、结果、缩略图、下载、播放清单和 MJPEG 流接口；检查请求参数与跨站写请求 | 调用 `TaskService`，用 `PreviewReader` 读取视频；服务退出时调用任务收尾 |
| `webapp/service.py` | `TaskService` 管理任务记录、独立流水线进程和停止信号；读取状态与耗时；按任务时间范围隔离视频；提取视频元数据、日志步骤并清理记录 | 启动根目录 `pipeline.py`；读取 `logs/`、缓存及标注视频目录 |
| `webapp/preview.py` | `PreviewReader` 在同一后台线程内打开、定位、解码和释放 OpenCV 视频；用容量为 4 的队列提前准备 JPEG 帧 | 为 `server.py` 的流播放提供帧，支持拖拽后的起始帧定位 |
| `webapp/static/index.html` | 页面中的创建任务、分析任务、标注画面、输出片段、耗时表和日志区域 | 加载 `style.css`、`app.js` 和 `player.js` |
| `webapp/static/style.css` | 颜色、布局、按钮、进度条、日志区域及窄屏适配 | 只影响页面显示 |
| `webapp/static/app.js` | 上传与启动任务、缓存模式选择、每 1.5 秒轮询、任务与设备切换、结果隔离、耗时和日志渲染 | 调用网站 API，将播放清单交给 `player.js`；切换视图时忽略旧请求返回 |
| `webapp/static/player.js` | 解析流中的 JPEG 帧与时间信息；控制进度条、暂停继续、拖拽及画面更新；释放旧图片 URL | 请求 `/api/stream/`，与 `app.js` 共用当前页面状态 |
| `webapp/__init__.py` | 将目录声明为 Python 包 | 供 `web.py` 导入 `webapp.server` |

启动及任务调用链：

```text
start_website.cmd
  → .venv/Scripts/python.exe web.py
  → webapp.server.create_app() → Uvicorn 提供网站服务

浏览器 index.html + app.js
  → POST /api/tasks
  → server.py → service.py 的 TaskService.start()
  → pipeline.py
  → VideoCapture/main.py + InsightfaceModel/main.py + VideoTag/main.py

浏览器 app.js / player.js
  → 任务、结果、日志、播放 API
  → service.py 读取本地记录和已生成视频
  → preview.py 解码 → server.py 输出帧 → player.js 显示
```

常用网站接口：

| 方法与路径 | 用途 |
| --- | --- |
| `GET /`、`GET /static/...` | 页面和静态资源 |
| `GET /api/overview` | 任务概览、设备列表、示例源 |
| `POST /api/uploads` | 上传源视频 |
| `POST /api/tasks` | 新建任务，参数含 `source`、`device_no`、`buffer_mode` |
| `GET /api/tasks/{identifier}` | 单任务状态、耗时和输出信息 |
| `POST /api/tasks/{identifier}/stop` | 停止任务 |
| `GET /api/tasks/{identifier}/logs` | 原始日志与处理步骤 |
| `DELETE /api/records` | 清理已结束任务记录 |
| `GET /api/results/{device_no}?job_id=...` | 当前任务片段列表 |
| `GET /api/results/{device_no}/{name}/thumbnail` | 视频缩略图 |
| `GET /api/results/{device_no}/{name}/download` | 下载标注 MP4 |
| `GET /api/playback/{device_no}?job_id=...` | 已生成片段的时长、偏移和可播放总长度 |
| `GET /api/stream/{device_no}?job_id=...&offset=...` | 顺序播放或定位到指定秒；未生成的位置拒绝访问 |

## 根目录文件的内容

| 文件 | 内容与用途 |
| --- | --- |
| `web.py` | 解析 `--host`、`--port`，创建网站应用并启动 Uvicorn。默认监听 `127.0.0.1:8080` |
| `start_website.cmd` | 切到脚本所在目录，检查 `.venv/Scripts/python.exe`，运行 `web.py`，退出后暂停窗口。不是三个模块独立 API 的启动脚本 |
| `pipeline.py` | 解析视频源、设备号和缓存模式；获取设备运行锁；分配本次时间范围；生成状态与任务配置；启动三个子进程并转发日志；处理停止、错误、排空及汇总 |
| `buffer_cleanup.py` | `BufferCleaner` 在运行模式下每 250 毫秒扫描所选设备的中间缓存；根据下游消费时间清理，重试占用文件；不删除标注视频 |
| `capture_buffers.py` | `second_directories()` 扫描秒目录，`ready_seconds()` 找到完成采集的秒，`buffer_usage()` 统计 JPEG 帧数和字节数；容忍下游同时删除文件 |
| `runtime_metrics.py` | `Metrics.measure()` 累计各步骤耗时和调用次数；`publish()` 发布阶段状态；`write_json()` 原子写 JSON，供网站和流水线读取 |
| `video_output.py` | `concatenate_segments()` 检查各片段分辨率、FPS 和帧数，用 FFmpeg 复制编码流拼接视频；支持取消，校验成片并清理临时拼接文件 |
| `requirements.txt` | 网站、视频读写、YOLO、InsightFace、ONNX Runtime、PyTorch 和 FFmpeg 的依赖声明 |
| `.gitignore` | 排除虚拟环境、模型、原帧、推理缓存、标注视频、日志和临时文件等，避免把运行产物提交到仓库 |
| `ATIO视频处理.drawio` | 项目设计图文件，可用 draw.io 打开，不参与程序执行 |
| `README.md` | 项目使用与开发说明，作为当前功能的统一文档入口 |

启动窗口按 Ctrl+C 可关闭网站服务，并等待任务收尾；网站“停止”按钮只停止对应任务。关闭浏览器不会停止服务。若提示端口 8080 已占用，说明已有服务在运行，本次启动失败，避免重复启动。

本地摄像头可在网站视频来源填写 `0`、`1` 等编号；这里选择的是运行 `web.py` 的电脑上的摄像头。“设备编号”是任务和输出目录的编号，与摄像头索引分开设置。

## 验证

| 文件 / 目录 | 职责 |
| --- | --- |
| `web.py`、`webapp/` | 网站、任务管理、播放器和预览 |
| `pipeline.py` | 子进程启停、任务范围和完成状态 |
| `VideoCapture/` | 全帧采集、源解析、重连 |
| `InsightfaceModel/` | 推理、跟踪、协议编码 |
| `VideoTag/` | 协议读取、逐帧标注、分段输出 |
| `buffer_cleanup.py`、`capture_buffers.py` | 动态清理、容错目录扫描 |
| `runtime_metrics.py`、`video_output.py` | 计时状态、完整视频拼接 |
| `tools/` | 隔离回归测试与浏览器验证 |

```powershell
.\.venv\Scripts\python.exe tools/pipeline_contract_smoke.py
.\.venv\Scripts\python.exe tools/performance_contract_smoke.py
.\.venv\Scripts\python.exe tools/tracker_identity_smoke.py
.\.venv\Scripts\python.exe tools/buffer_cleanup_smoke.py
.\.venv\Scripts\python.exe tools/pipeline_smoke.py --seconds 30 --source "D:\video.mp4" --buffer-mode runtime
.\.venv\Scripts\python.exe tools/pipeline_smoke.py --seconds 3 --source "D:\video.mp4" --buffer-mode test
```

真实模型测试在 `logs/` 创建隔离目录，不消费已有缓存。`pipeline_lifecycle_smoke.py` 验证停止与故障收尾；`playback_contract_smoke.py` 验证帧顺序、播放节奏和记录清理，需要测试依赖 `httpx`。

浏览器测试需 Node.js、Chrome 和 Playwright：

```powershell
npm install --prefix logs/web_browser_qa playwright
.\.venv\Scripts\python.exe tools/web_smoke.py
node tools/page_session_smoke.cjs
node tools/player_seek_smoke.cjs
node tools/result_isolation_smoke.cjs
```

浏览器脚本使用 Windows Chrome 默认安装路径，迁移环境需调整 `executablePath`。网站本身不依赖 Node.js 或 Playwright。
