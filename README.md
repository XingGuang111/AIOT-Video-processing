# AIOT 视频结构化分析系统

基于 **YOLOv8s 人体检测 + ByteTrack 跟踪 + InsightFace 人脸特征** 的视频结构化分析流水线。三个模块以**独立进程 + 磁盘目录**解耦协作，可单独启停、单独调试、单独替换算法。

---

## 0、基础知识和原理

### yolov8目标检测算法

设计CNN卷积神经网络

![image-20260701103335992](assets/image-20260701103335992.png)

 六元组

yolov8是一个相当复杂的卷积神经网络
 将输入图像计算成特征图→3组特征值图

![image-20260701103322317](assets/image-20260701103322317.png)

![image-20260701103420662](assets/image-20260701103420662.png)



 训练模型:yolov8s.pt

    它是 **YOLOv8s 的训练好的模型权重文件**,里面包含:

- 网络结构 + 训练好的参数(在 COCO 数据集上训练,能识别 80 类物体);
- person(人)是其中的第 0 类(COCO id=0)。

`ultralytics` 库把这个 `.pt` 加载进来,就能对图像做推理输出检测框。

###   insightface

人脸识别开源仓库

![image-20260702095853258](assets/image-20260702095853258.png)

- |          | 人体检测           | 人脸检测 + 特征               |
  | -------- | ------------------ | ----------------------------- |
  | 库       | ultralytics (YOLO) | insightface                   |
  | 模型文件 | `yolov8s.pt`       | `buffalo_l/` 里的一堆 `.onnx` |
  | 框架     | PyTorch (torch)    | ONNX Runtime                  |
  | 输出     | 人体框             | 人脸框 + 512 维特征           |

`buffalo_l`是一个"模型包",目录下包含多个 `.onnx` 文件,各管一件事:

- `det_10g.onnx` —— **人脸检测**(SCRFD 算法),找出脸在哪;
- `w600k_r50.onnx` —— **人脸识别**(ArcFace 算法),输出那 512 维特征向量;
- `2d106det.onnx` / `1k3d68.onnx` —— 关键点(五官定位);
- `genderage.onnx` —— 性别年龄(本项目没用到)。

### ByteTrack 多目标跟踪

ByteTrack是一种基于Tracking-by-Detection范式的多目标跟踪算法。
 ByteTrack解决了多目标跟踪中的一个重要问题，即**低分检测框的处理**。
 传统的跟踪方法通常只关联高分检测框，而低分检测框通常被忽略，这会导致一些真实的目标被漏检或者
 跟踪轨迹被碎片化。
 ByteTrack通过关联几乎每个检测框，包括低分检测框，利用它们与轨迹片段的相似性来恢复真实目标并
 过滤掉背景检测，从而提高了检测器的性能。

将不同的照片中,相互关联。

---

## 一、系统架构与数据流

### 1.1 完整数据流

```
VideoCapture          InsightfaceModel              VideoTag
（采集）        →    （AI 推理）            →     （标注渲染）
     │                    │                          │
 capture_buffer/      101_buffer/ 201_buffer/     annotated_buffer/
 设备/时间戳/*.jpg     设备/{ts}.bin 协议           设备/{ts}_{ts}.mp4
```

1. **VideoCapture**：从本地视频 / RTSP / USB 摄像头读取画面 → 编码 JPEG → 按"设备号/秒级时间戳/帧序号"三级目录输出到 `capture_buffer/`；
2. **InsightfaceModel**：轮询读取帧 → 人体检测 + 跟踪 + 人脸特征 → 每秒编码 `101`（人体）、`201`（人脸）二进制协议落盘；
3. **VideoTag**：读取协议 + 原始视频 → 按时间戳关联目标 → 绘制标注框 → 分段输出 MP4 视频（默认 `mp4v` 编码，可配 `avc1`/H.264）；
4. 模块间**只通过磁盘目录交换数据**，无代码级依赖。

### 1.1.1 架构示意图

```mermaid
flowchart LR
    subgraph VC["VideoCapture（采集）"]
        cam[摄像头/视频源/RTSP] --> jpg["抽帧 JPEG<br/>device_N/&lt;ts&gt;/frame_*.jpg"]
    end
    subgraph IM["InsightfaceModel（AI 推理）"]
        det["YOLOv8s 人体检测<br/>+ ByteTrack 跟踪"] --> face["InsightFace<br/>整帧人脸 512 维特征"]
        face --> enc["协议编码<br/>CRC32 + 原子写"]
    end
    subgraph VT["VideoTag（标注渲染）"]
        dec["协议解码校验<br/>按 ts 重组"] --> draw["逐帧绘框<br/>绿:人体 红:人脸 白:ROI"]
        draw --> wr["分段 MP4 编码"]
    end

    jpg -->|"capture_buffer/<br/>device_N/&lt;ts&gt;/*.jpg"| det
    enc -->|"101_buffer + 201_buffer/<br/>device_N/{ts}.bin"| dec
    src[(原始视频)] --> draw
    wr -->|"annotated_buffer/<br/>device_N/{ts}_{ts}.mp4"| out[标注视频]

    classDef buf fill:#eef,stroke:#88a;
    class jpg,enc,wr buf;
```

> 三模块为独立进程，箭头上的磁盘目录是唯一的数据交换边界；任一模块可单独启停、替换算法。

### 1.2 项目目录结构

```
1/  (项目根)
├── capture_buffer/          # VideoCapture 输出帧（InsightfaceModel 只读）
│   └── device_N/<ts>/frame_*.jpg
├── 101_buffer/              # 人体检测协议    device_N/{ts}.bin
├── 201_buffer/              # 人脸特征协议    device_N/{ts}.bin
├── annotated_buffer/        # 标注视频        device_N/{ts}_{ts}.mp4
├── config/
│   ├── config.yaml          # 全局统一配置（两模块共享）
│   └── models/              # 模型权重：yolov8s.pt、buffalo_l/（首次自动下载）
├── logs/<YYYYMMDD>/         # capture.log / inference.log / tag.log
├── VideoCapture/            # 视频帧采集模块
│   ├── app.py               # FastAPI 服务入口（端口 8000）
│   ├── main.py              # CLI 入口：server / extract
│   ├── config.yaml          # VideoCapture 独立配置
│   ├── videocapture/        # 核心包（manager/worker/cleaner/parser）
│   └── demo_videos/         # 示例视频：video1.mp4、video2.mp4、video3.mp4
├── InsightfaceModel/        # AI 推理模块
├── VideoTag/                # 视频标注模块
├── requirements.txt         # 根依赖（InsightfaceModel / VideoTag）
└── README.md
```

> 三个子模块各有自己的 `requirements.txt`：`VideoCapture/requirements.txt`（FastAPI 采集）、`InsightfaceModel/requirements.txt`（如未提供则用根 `requirements.txt`）、`VideoTag/` 同理。`VideoCapture` 与下游模块**不共享 Python 虚拟环境**（前者是 FastAPI 服务，后两者是 OpenCV + 模型推理），可各自独立安装。

> **⚠️ Demo 视频不入仓**：`VideoCapture/demo_videos/` 下的 `video1.mp4` / `video2.mp4` / `video3.mp4` 已加入 `.gitignore`（体积约 180MB）。如需复现 README 中的示例命令，请将本地测试视频放入 `VideoCapture/demo_videos/` 目录，或将命令中的 `VideoCapture/demo_videos/video1.mp4` 替换为你的实际视频路径。

---

## 二、快速开始（端到端使用案例）

### 2.1 安装依赖

三模块的依赖各自独立，按需安装即可。

```bash
# 建议根目录虚拟环境
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 1) VideoCapture（采集，FastAPI）
pip install -r VideoCapture/requirements.txt

# 2) InsightfaceModel + VideoTag（推理+标注，OpenCV/YOLO/InsightFace）
pip install -r requirements.txt
```

> 国内网络慢可加镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
> 首次运行 InsightfaceModel 会自动下载 `yolov8s.pt` 与 `buffalo_l` 到 `config/models/`。

### 2.2 准备输入帧（采集）

> **核心原则：原视频是多少帧就多少帧处理，不要降采样。** 这样下游推理才能拿到最多的人体/人脸轨迹细节，标注视频也最贴近原始画面。VideoCapture 通过 `sampling=-1` 实现全量抽帧（每帧都写盘），文件模式会按原始 FPS 限速以模拟真实播放。

VideoCapture 提供两种使用方式：**API 服务模式**（多路并发，远程管控）和 **CLI 单路模式**（临时抽取）。详细参数、接口、示例见 [第三章 VideoCapture](#三videocapture--视频帧采集模块)。

- **方式一（推荐，API 服务模式）**：启动 `python VideoCapture/main.py server` → POST `/api/v1/capture/start` 启动采集（关键参数 `sampling=-1` 全量抽帧），POST `/api/v1/capture/stop` 停止，GET `/api/v1/capture/status` 查状态；

- **方式二（CLI 单路模式）**：直接 `python VideoCapture/main.py extract <视频源>`，按 `Ctrl+C` 停止。

  ```powershell
  # 默认 sampling=-1 全量抽帧（每帧都写盘，文件模式按原视频 FPS 限速）
  python VideoCapture/main.py extract .\VideoCapture\demo_videos\video3.mp4

  # 显式每秒抽 1 帧（降采样，调试或磁盘紧张时用）
  python VideoCapture/main.py extract .\VideoCapture\demo_videos\video3.mp4 --sampling 1
  ```

  CLI `extract` 子命令默认 `sampling=-1`（全量抽帧），可通过 `--sampling N` 改为每 N 秒抽 1 帧。

输出的帧会落在 `capture_buffer/device_N/<秒级时间戳>/frame_XXXXXX.jpg`，供下游模块读取。

> 也可以**手动放置帧**到 `capture_buffer/device_1/<秒级时间戳>/frame_0001.jpg ...`（每个秒目录代表 1 秒），跳过采集步骤直接进入推理。

### 2.3 运行推理（生成协议）

> **关键参数 `--fps` 必须与原视频帧率一致**，否则会出现"采集写了 N 帧但推理只抽 N/3 帧"或"推理抽的帧不在采集秒目录里"的情况，导致协议时间戳与原始视频对不齐。

demo 视频原始帧率：video1.mp4 = **30 fps**、video2.mp4 = **60 fps**、video3.mp4 = **18 fps**。确认方式：

```bash
python -c "import cv2; c=cv2.VideoCapture('VideoCapture/demo_videos/video3.mp4'); print(c.get(cv2.CAP_PROP_FPS))"
```

```bash
# video3.mp4 原始 18 fps，所以这里传 18
python InsightfaceModel/main.py --device-no device_0 --fps 18
```

```bash
# video1.mp4 原始 30 fps
python InsightfaceModel/main.py --device-no device_0 --fps 30
```

```bash
# video2.mp4 原始 60 fps（4K 推理较慢，必要时可降到 30 折中）
python InsightfaceModel/main.py --device-no device_0 --fps 60
```

日志持续输出 `ts=... 人体=.. 人脸=..`，并在 `101_buffer/device_0/`、`201_buffer/device_0/` 生成 `.bin`；连续无新帧达 `idle_timeout` 秒后自动回收线程退出。

### 2.4 运行标注（生成视频）

```bash
python VideoTag/main.py --source "./VideoCapture/demo_videos/video3.mp4" --device-no device_0
```

输出 `annotated_buffer/device_0/{起始ts}_{结束ts}.mp4`，默认每 60 秒一段。

### 2.5 API 方式（可选，远程管控）

```bash
# VideoCapture 采集（8000 端口，已在 2.2 启动）
python InsightfaceModel/api.py --port 5001   # 推理服务
python VideoTag/api.py        --port 5003   # 标注服务
```

调用见 [第八章](#八http-api-使用说明)。

### 2.6 完整操作流程（Windows PowerShell）

> 关键顺序：**采集 → 必须先推理（生成 101/201 协议）→ 再标注（消费协议）**。顺序反了标注会因无协议数据画不出框。

#### 方式一：CLI 直接跑（推荐，单终端按序执行）

```powershell
# 第 1 步：进目录并激活虚拟环境（每个新终端都要）
cd "D:\笔记\python程序\1\1"
.\.venv\Scripts\Activate.ps1

# 第 2 步：采集（API 方式）—— sampling=-1 全量抽帧，保留原视频每一帧
# 终端 A：启动采集服务
python VideoCapture/main.py server
# 终端 B：POST 启动采集（单路视频1，全量抽帧；按原视频 30 fps 限速）
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video1.mp4","sampling":-1,"jpg_quality":75}'

# 第 3 步：跑推理（--fps 必须与原视频帧率一致：video1=30、video2=60、video3=18）
python InsightfaceModel/main.py --device-no device_0 --fps 30

# 第 4 步：跑标注（读协议 + 视频 → 生成 MP4）
python VideoTag/main.py --source "VideoCapture/demo_videos/video1.mp4" --device-no device_0

# 第 5 步：查看结果
explorer annotated_buffer\device_0

# 采集中途可随时停：POST /api/v1/capture/stop {"device_no":0}；或采集端 Ctrl+C
```

CLI 单路速跑版（不用启服务，适合一次性抽帧，CLI 默认就是全量抽帧）：

```powershell
# 第 2 步替代：CLI 直接抽帧（前台运行，按 Ctrl+C 停止；文件模式按原视频 FPS 限速）
python VideoCapture/main.py extract "VideoCapture/demo_videos/video1.mp4"
```

#### 方式二：API（需 4 个终端，api.py 会占用终端）

```powershell
# 终端 1 —— 采集服务
cd "D:\笔记\python程序\1\1"; .\.venv\Scripts\Activate.ps1
python VideoCapture/main.py server
# 服务起来后访问 http://127.0.0.1:8000/docs 看接口文档

# 终端 2 —— 推理服务
cd "D:\笔记\python程序\1\1"; .\.venv\Scripts\Activate.ps1
python InsightfaceModel/api.py --port 5001

# 终端 3 —— 标注服务
cd "D:\笔记\python程序\1\1"; .\.venv\Scripts\Activate.ps1
python VideoTag/api.py --port 5003

# 终端 4（不用在本地） —— 按顺序发请求
# (1) 启动采集（视频1，原视频 30 fps，sampling=-1 全量抽帧）
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video1.mp4","sampling":-1,"jpg_quality":75}'
# (2) 查采集状态
Invoke-RestMethod http://127.0.0.1:8000/api/v1/capture/status
# (3) 启动推理（fps 必须等于原视频 30）
Invoke-RestMethod -Uri http://127.0.0.1:5001/api/v1/inference/start -Method Post -ContentType "application/json" -Body '{"fps":30,"device_no":"device_0"}'
# (4) 查推理状态，等 alive 变 false 表示推理处理完
Invoke-RestMethod http://127.0.0.1:5001/api/v1/inference/status
# (5) 推理完成后启动标注
Invoke-RestMethod -Uri http://127.0.0.1:5003/api/v1/tag/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video1.mp4","device_no":"device_0"}'
# (6) 查标注状态
Invoke-RestMethod http://127.0.0.1:5003/api/v1/tag/status
# (7) 停止采集
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/stop -Method Post -ContentType "application/json" -Body '{"device_no":0}'

# 视频3长视频示例（原视频 18 fps，sampling=-1 全量抽帧）
# (1) 启动采集
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video3.mp4","sampling":-1,"jpg_quality":75}'
# (2) 查采集状态
Invoke-RestMethod http://127.0.0.1:8000/api/v1/capture/status
# (3) 启动推理（fps 必须等于原视频 18）
Invoke-RestMethod -Uri http://127.0.0.1:5001/api/v1/inference/start -Method Post -ContentType "application/json" -Body '{"fps":18,"device_no":"device_0"}'
# (4) 查推理状态
Invoke-RestMethod http://127.0.0.1:5001/api/v1/inference/status
# (5) 启动标注
Invoke-RestMethod -Uri http://127.0.0.1:5003/api/v1/tag/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video3.mp4","device_no":"device_0"}'
# (6) 查标注状态
Invoke-RestMethod http://127.0.0.1:5003/api/v1/tag/status
# (7) 停止采集
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/stop -Method Post -ContentType "application/json" -Body '{"device_no":0}'

# 视频2示例（原视频 60 fps，4K 推理较慢，必要时可降到 30 折中）
# (1) 启动采集
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video2.mp4","sampling":-1,"jpg_quality":75}'
# (3) 启动推理（fps 60 或 30 折中）
Invoke-RestMethod -Uri http://127.0.0.1:5001/api/v1/inference/start -Method Post -ContentType "application/json" -Body '{"fps":60,"device_no":"device_0"}'
# (5) 启动标注
Invoke-RestMethod -Uri http://127.0.0.1:5003/api/v1/tag/start -Method Post -ContentType "application/json" -Body '{"source":"VideoCapture/demo_videos/video2.mp4","device_no":"device_0"}'
# (7) 停止采集
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/capture/stop -Method Post -ContentType "application/json" -Body '{"device_no":0}'
```

> **PowerShell 注意**：`curl` 在 PS 中是 `Invoke-WebRequest` 别名，语法与真 curl 不同；要用真 curl 请写 `curl.exe`。
> **浏览器注意**：监听地址 `0.0.0.0` 不可在浏览器打开，要用 `127.0.0.1`；根路径 `/` 无页面（404 正常）；`start`/`stop` 是 POST，浏览器地址栏只能 GET，只有 `/status` 能直接在浏览器打开验活。VideoCapture 的 API 文档页 `http://127.0.0.1:8000/docs` 可以直接浏览器打开（FastAPI Swagger UI）。

---

## 三、VideoCapture —— 视频帧采集模块

### 3.1 职责与输入输出

- **职责**：从本地视频文件 / RTSP 流 / USB 摄像头 / HTTP 流读取画面，编码为 JPEG，按"设备号/秒级时间戳/帧序号"三级目录输出。**不做检测/识别/告警**，只稳定、连续、可追溯地落盘。
- **输入**：`source`（文件路径 / RTSP URL / 纯 IP / 数字摄像头号 / HTTP URL）
- **输出**：`capture_buffer/device_N/<unix_ts>/frame_XXXXXX.jpg`（默认输出目录 `capture_buffer`，可在 `VideoCapture/config.yaml` 配 `global.output_dir`）
- **日志**：`VideoCapture/logs/<date>/capture.log`

### 3.2 分层结构

```
VideoCapture/
├── app.py               # FastAPI 路由层（端口 8000）：/capture/start、/capture/stop、/capture/status
├── main.py              # CLI 入口：python main.py server | extract <source>
├── config.yaml          # VideoCapture 独立配置（与 InsightfaceModel 不共享）
├── videocapture/        # 核心包
│   ├── capture_manager.py   # 多路采集管理器：设备编号分配 / 启停 / 状态查询
│   ├── capture_worker.py    # 单路采集线程：cv2.VideoCapture + 编码 + 写盘 + 重连
│   ├── buffer_cleaner.py    # 后台线程：按 max_seconds/frames/disk 清理旧帧
│   ├── source_parser.py     # 视频源类型识别（file/rtsp/device/stream）+ 海康 RTSP URL 拼接
│   ├── models.py            # CaptureStatusEnum 状态枚举 + CaptureStatus 数据类
│   └── logger.py            # 统一日志，RTSP 密码脱敏
├── demo_videos/         # 示例视频：video1.mp4 / video2.mp4 / video3.mp4
└── logs/                # 运行日志
```

### 3.3 工作流程

1. **源类型识别**（`source_parser`）：按字符串前缀匹配判定 `device` / `rtsp` / `stream` / `file`，纯 IPv4 自动按海康默认规则拼接 RTSP URL。
2. **多路调度**（`CaptureManager`）：每个 source 起一个 `CaptureWorker` 线程，自动分配递增的 `device_no`，线程安全的状态查询/启停。
3. **采集循环**（`CaptureWorker`）：`cv2.VideoCapture(source)` → 读帧 → `cv2.imencode('.jpg')` → 按秒级时间戳建目录写盘；文件模式按原始 FPS 限速以模拟真实播放。
4. **缓存控制**：每 30 帧检查 `max_frames` / `max_disk_mb`；`BufferCleaner` 后台线程定时检查 `max_seconds` / `total_max_disk_mb`，超限按秒目录时间戳升序淘汰。
5. **RTSP 重连**：单次读帧失败不退出，连续失败达 `max_frame_fails` 触发重连，超过 `max_reconnect_times` 标记 `error`；停止信号优先级最高。
6. **日志脱敏**：RTSP 密码在日志中替换为 `***`。

#### 采集时序

```mermaid
sequenceDiagram
    participant U as 用户/上游
    participant A as app.py(FastAPI)
    participant M as CaptureManager
    participant W as CaptureWorker
    participant CV as cv2.VideoCapture
    participant D as capture_buffer
    participant C as BufferCleaner

    U->>A: POST /capture/start {source, ...}
    A->>M: start(source, ...)
    M->>M: 分配 device_no
    M->>W: 启动线程
    loop 采集循环
        W->>CV: read()
        CV-->>W: frame
        W->>W: cv2.imencode → JPEG
        W->>D: 写 device_N/<ts>/frame_XXXXXX.jpg
        W->>W: 每 30 帧检查 max_frames / max_disk_mb
    end
    Note right of C: 后台线程<br/>每 N 秒检查 max_seconds / total_max_disk_mb
    C->>D: 超限删最旧秒目录
    U->>A: POST /capture/stop {device_no}
    A->>M: stop(device_no)
    M->>W: stop_event.set()
    W-->>M: 线程退出
```

### 3.4 启动方式

#### 方式一：API 服务模式（推荐，支持多路并发）

```bash
cd VideoCapture
python main.py server
# 启动后访问接口文档：http://127.0.0.1:8000/docs
```

启动后通过 HTTP 接口管理：

```bash
# 启动单路采集
curl -X POST http://127.0.0.1:8000/api/v1/capture/start \
  -H "Content-Type: application/json" \
  -d '{"source":"demo_videos/video1.mp4","sampling":1,"jpg_quality":75}'

# 启动多路（视频 + USB 摄像头）
curl -X POST http://127.0.0.1:8000/api/v1/capture/start \
  -H "Content-Type: application/json" \
  -d '{"source":["demo_videos/video1.mp4","0"],"sampling":1,"max_seconds":300}'

# 查询状态
curl http://127.0.0.1:8000/api/v1/capture/status

# 停止指定设备
curl -X POST http://127.0.0.1:8000/api/v1/capture/stop \
  -H "Content-Type: application/json" \
  -d '{"device_no":0}'

# 停止全部
curl -X POST http://127.0.0.1:8000/api/v1/capture/stop \
  -H "Content-Type: application/json" -d '{}'
```

Windows PowerShell 把 `curl` 换成 `Invoke-RestMethod` 即可（见 2.6 方式二示例）。

#### 方式二：CLI 单路模式（前台运行）

无需启动服务，命令行直接抽帧，**适合一次性测试**。`Ctrl+C` 停止；文件模式播放完毕自动退出。

```bash
cd VideoCapture

# 本地视频文件（默认 sampling=-1 全量抽帧，每帧都写盘，文件模式按原 FPS 限速）
python main.py extract demo_videos/video1.mp4

# 显式每秒抽 1 帧（降采样，调试 / 磁盘紧张时用）
python main.py extract demo_videos/video1.mp4 --sampling 1

# USB 摄像头（设备号 0），30 秒自动停止
python main.py extract 0

# RTSP 摄像头（已知完整地址）
python main.py extract rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101

# RTSP 摄像头（纯 IP，自动拼接海康默认 URL）
python main.py extract 192.168.1.64
# 注意：纯 IP 必须配账号密码（CLI 模式不直接支持 → 用 API 方式）
```

> CLI 模式的 `extract` 支持 `--sampling` 调抽帧密度；RTSP 账号密码/采样间隔之外的高级参数（多路并发、device_no 指定、jpg_quality 等）请用 API 方式。

### 3.5 视频源类型识别规则

`source_parser.parse_source()` 按字符串前缀匹配判定类型，**不做网络探测**：

| 输入示例 | 识别类型 | 处理方式 |
|----------|----------|----------|
| `0`, `1`, `2` | `device` | `cv2.VideoCapture(设备号)` → USB 摄像头 |
| `rtsp://admin:pwd@ip:554/...` | `rtsp` | 直接作为 VideoCapture 输入 |
| `192.168.1.64` | `rtsp` | 拼接 `rtsp://account:password@ip:554/Streaming/Channels/101`（需 `account`/`password`） |
| `http://server/video.mp4` | `file` | 扩展名命中视频，按文件 URL 打开（视为文件模式） |
| `http://ip/live/stream` | `stream` | 非视频扩展名，按网络流处理，失败重试 |
| `demo/video1.mp4` | `file` | 兜底匹配，按本地文件打开 |

视频扩展名集合：`{.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v, .ts}`。

### 3.6 关键参数（API 请求体）

`POST /api/v1/capture/start` 入参：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `source` | string \| string[] | 是 | — | 字符串=单路，列表=多路 |
| `account` | string | 否 | null | RTSP 账号（**纯 IP 时必填**） |
| `password` | string | 否 | null | RTSP 密码（**纯 IP 时必填**） |
| `device_no` | int | 否 | null | 设备编号；为空时自动递增分配（从 0 起） |
| `output_dir` | string | 否 | config 中 `global.output_dir` | 输出根目录；最终路径为 `<output_dir>/device_N/...` |
| `jpg_quality` | int | 否 | 75 | JPEG 质量 1-100（影响磁盘占用与清晰度） |
| `sampling` | float | 否 | 1.0 | 采样间隔（秒）；`-1` = 全量提取（每帧都写，**推荐：原帧率处理**） |
| `max_seconds` | int | 否 | null | 最大采集时长（秒）；`0` = 不限制；超时自动 `finished` |
| `max_frames` | int | 否 | null | 单路最大帧数；超出后删最旧秒目录 |
| `max_disk_mb` | int | 否 | null | 单路最大磁盘占用（MB）；超出后删最旧秒目录 |

### 3.7 输出目录结构

帧文件按 **设备号 → Unix 时间戳（秒）→ 帧序号** 三级目录组织：

```
capture_buffer/
├── device_0/                    ← 第 1 路视频源
│   ├── 1717891200/              ← 第 1 秒
│   │   ├── frame_000000.jpg
│   │   └── frame_000001.jpg
│   └── 1717891201/              ← 第 2 秒
│       └── frame_000000.jpg
└── device_1/                    ← 第 2 路视频源
    └── 1717891200/
        └── frame_000000.jpg
```

- **实时流（RTSP/USB）**：时间戳取系统当前时间；
- **文件模式**：时间戳基于视频播放进度推算，并按原始 FPS 限速模拟真实播放。

### 3.8 配置说明（VideoCapture/config.yaml）

```yaml
global:
  output_dir: "capture_buffer"   # 输出根目录，相对项目根解析
  default_fps: 25                 # 视频源 FPS 不可用时回退值

reconnect:
  max_reconnect_times: 10         # RTSP 最大重连次数
  reconnect_interval: 1           # 重连间隔（秒）
  max_frame_fails: 5              # 连续读帧失败阈值（触发重连）

buffer_clean:
  max_seconds: 600                # 保留最近 N 秒（0=不限制）
  max_frames: 0                   # 单路最大帧数（0=不限制）
  max_disk_mb: 2048               # 单路最大磁盘占用 MB（0=不限制）
  total_max_disk_mb: 10240        # 全局最大磁盘占用 MB（0=不限制）
  clean_interval: 15              # 清理检查间隔（秒）
```

> **路径解析**：`output_dir` 相对**项目根目录**（`1/1/`）解析，与运行 `VideoCapture/main.py` 时所在目录无关；当前默认 `capture_buffer`，与 InsightfaceModel 约定的输入目录一致，无需改。
> **采集循环内的检查**（`max_frames` / `max_disk_mb`）每 30 帧触发一次；**BufferCleaner 的检查**（`max_seconds` / `total_max_disk_mb`）每 `clean_interval` 秒触发一次。

### 3.9 常见场景示例（API 请求体）

> **默认推荐 `sampling=-1` 全量抽帧**，把原视频每一帧都写盘，下游推理按原视频帧率处理，结果最好；只有磁盘紧张或调试时再用 `sampling=1`（每秒 1 帧）等降采样档位。

| 场景 | source | 关键参数 |
|------|--------|---------|
| **本地视频全量抽帧（推荐）** | `"VideoCapture/demo_videos/video1.mp4"` | `sampling=-1, jpg_quality=75` |
| 本地视频每秒抽 1 帧（降采样） | `"VideoCapture/demo_videos/video1.mp4"` | `sampling=1, jpg_quality=75` |
| 两个视频同时转（全量） | `["video1.mp4","video2.mp4"]` | `sampling=-1`（输出 `device_0/`、`device_1/`） |
| 摄像头限时限量 | `"0"` | `max_seconds=60, max_frames=500, max_disk_mb=50` |
| 视频 + 摄像头一起 | `["video1.mp4","0"]` | `sampling=-1, max_seconds=300` |
| RTSP（已知完整 URL） | `"rtsp://admin:12345@192.168.1.64:554/..."` | `max_seconds=300` |
| RTSP（纯 IP，自动拼接） | `"192.168.1.64"` | `account="admin", password="12345", max_seconds=300` |

### 3.10 状态查询与状态机

`GET /api/v1/capture/status?device_no=N` 返回字段：

```json
{
  "code": 0,
  "data": {
    "device_no": 0,
    "source": "0",
    "source_type": "device",
    "fps": 30.0,
    "frame_count": 900,
    "resolution": "640x480",
    "frames_captured": 150,
    "status": "running",
    "last_error": "",
    "start_time": 1717891200.0
  }
}
```

状态枚举：

| 状态 | 含义 |
|------|------|
| `initialized` | 已初始化（线程已起但还没读到第一帧） |
| `running` | 采集中 |
| `finished` | 文件播放完毕 / 达到 `max_seconds` 自动停止 |
| `error` | 异常（如 RTSP 重连超限） |
| `reconnecting` | RTSP 断线重连中 |
| `stopped` | 手动停止 |

### 3.11 核心设计要点

- **文件模式限速**：`sampling=-1` 全量提取时按视频原始 FPS 限速，避免快速抽空文件破坏下游时序。
- **三级缓存限制**：单路 `max_frames` / `max_disk_mb`（采集循环检查，30 帧一次）+ 全局 `total_max_disk_mb`（BufferCleaner 定时检查）；淘汰策略一律按秒目录 Unix 时间戳升序删最旧。
- **RTSP 断线重连**：单次读帧失败不立即退出，等待重试；连续失败达 `max_frame_fails` 触发重连；超过 `max_reconnect_times` 标记 `error`；停止信号优先级最高。
- **日志安全**：`desensitize_rtsp_url()` 将 RTSP 密码替换为 `***`，不会明文输出摄像头密码。
- **目录解耦**：输出目录 `capture_buffer/` 是与 InsightfaceModel 的**唯一**数据交换边界，下游只读不写。

### 3.12 VideoCapture 依赖

`VideoCapture/requirements.txt`：

```
opencv-python>=4.5.5.62     # 视频读取与帧编码
fastapi>=0.100.0            # API 服务框架
uvicorn>=0.23.2             # ASGI 服务器
pyyaml>=6.0                 # 配置解析
pydantic>=2.0.0             # 请求体校验
python-multipart>=0.0.6     # 文件上传支持
```

> VideoCapture 与下游（InsightfaceModel/VideoTag）**不共享虚拟环境**也可独立运行；只跑采集/只跑标注/只跑推理都行。

---

## 四、InsightfaceModel —— AI 推理模块

### 4.1 输入输出

- **输入**：`capture_buffer/device_N/<ts>/*.jpg`
- **输出**：`101_buffer/device_N/{ts}.bin`、`201_buffer/device_N/{ts}.bin`、`logs/<date>/inference.log`

### 4.2 分层结构

```
InsightfaceModel/
├── api.py               # API 层：Flask HTTP（5001），仅调用服务层
├── main.py              # 服务层：CLI 入口、设备调度
├── manager.py           # 服务层：多设备线程管理（Manager/Worker + Event）
├── config_loader.py     # 配置层：加载校验 config.yaml
├── logger.py            # 基础层：统一日志，按日期输出 logs/<date>/inference.log
├── inference_engine.py  # 基础层：PersonDetector(YOLOv8) + FaceEmbedder(InsightFace) 薄封装
├── tracker.py           # 引擎层：ByteTrack 记账薄封装（last_seen / reset）
├── model_processor.py   # 引擎层：帧处理主流程
└── protocol_encoder.py  # 数据层：101/201 编码、CRC32、原子写
```

### 4.3 工作流程

1. **轮询**：扫描 `capture_buffer/device_N/` 下新增的秒级目录，按时间戳升序处理（保证 ByteTrack 时序）。
2. **抽帧**：每个秒目录按 `fps` 等间隔抽帧，`imdecode` 失败的损坏/空帧跳过。
3. **人体检测 + 跟踪**：YOLOv8s 仅取 person 类（COCO id=0），ultralytics 内置 ByteTrack 输出稳定 `track_id`；按 `person_min_pixels`、`edge_margin`、`person_min_top_margin` 过滤无效框。
   - **`person_min_top_margin`**：人体框顶部距画面顶部 < 此值视为"头部被裁掉只露下半身"（如低角度门口摄像头人员刚进入的瞬间），直接丢弃——避免分配 track_id 后画面出现"只有绿框、无 F 框"的鬼影，也避免 ID 浪费；蹲姿/坐姿的人 bbox top 在画面中下部，**不会被误杀**。
4. **人脸特征**：对**整帧**调用 InsightFace `buffalo_l`，得到每张脸的框、置信度、512 维 `normed_embedding`。
5. **关联**：`person_id` = 人脸框中心落入的人体框 `track_id`（无则 `NO_PERSON_ID=0xFFFFFFFF`）；`face_id` = `person_id`（跟随人稳定，红框 ID 与绿框 ID 一致）。
6. **人脸去重**（关键，避免"一张脸多个红框"）：
   - **帧内 NMS**（IoU>0.5）：InsightFace 同帧可能对同一张脸输出多个高度重叠的框；
   - **跨帧 NMS**（IoU>0.3）：同秒内多帧检测到的同一张脸合并为一个；
   - **按 `person_id` 聚合**：每人在该秒内只保留 conf 最高的 1 张脸（解决 InsightFace 把头部/肩膀/衣服纹理误检为多张脸的问题——它们的 IoU 都不高无法用 NMS 合并，但都关联到同一 `person_id`）。
7. **落盘**：每秒聚合 → 编码 101/201 → `.tmp` 写完 `os.replace` 原子重命名。
7. **空闲回收**：连续 `idle_timeout` 秒无新帧，释放模型/GPU 并退出。

#### 处理时序

```mermaid
sequenceDiagram
    participant M as ModelProcessor
    participant B as capture_buffer
    participant Y as PersonDetector(YOLO+ByteTrack)
    participant F as FaceEmbedder(InsightFace)
    participant P as protocol_encoder

    loop 每秒目录（ts 升序）
        M->>B: 扫描新增 <ts>/ 目录
        M->>M: 按 fps 抽帧、过滤坏帧
        loop 每帧
            M->>Y: detect(frame)
            Y-->>M: persons[{track_id, xyxy, conf}]
            M->>F: embed(整帧)
            F-->>M: faces[{xyxy, conf, embedding}]
            M->>M: 帧内 NMS(IoU>0.5)
            Note right of M: person_id=框中心落入的 track_id
        end
        M->>M: 跨帧 NMS(IoU>0.3) → 按 person_id 聚合（每人留 1 张 conf 最高）
        Note right of M: face_id = person_id<br/>(红框 ID 跟随人稳定)
        M->>P: encode_101 / encode_201（每秒聚合）
        P->>P: CRC32 + 写 .tmp → os.replace
    end
    Note over M: 空闲 idle_timeout 秒无新帧 → 释放模型/GPU 退出
```

### 4.4 ID 语义（重要）

| ID | 含义 | 稳定性 |
|---|---|---|
| `track_id`（绿框 `ID:X`） | ByteTrack 人体跟踪 ID | 同一人持续在画面/短时遮挡内**保持不变**；离开超过 `track_buffer` 帧（默认 30，10fps≈3 秒）后再入镜分配**新号**，号只增不复用 |
| `face_id`（红框 `F:X`） | **与 `person_id` 同值**，即该人脸所属人体的 `track_id` | 跨帧稳定，**跟随人**；未关联到人体时为 `NO_PERSON_ID (0xFFFFFFFF)`，红框只画不显示 label |
| `person_id`（201 字段） | 人脸所属人体 | = 该脸所在人体框的 `track_id`，跨帧稳定 |

> **红框 ID 与绿框 ID 保持一致**（都是 `track_id`），实现"红框跟随人的运动而动"。实现机制：
> 1. 帧内 NMS（IoU>0.5）合并同帧 InsightFace 冗余检测框；
> 2. 跨帧 NMS（IoU>0.3）合并同秒内多帧重复检测的同一张脸；
> 3. **按 `person_id` 聚合**，每人在该秒内只保留 conf 最高的 1 张脸（关键：InsightFace 在低质量/复杂背景下常把头部/肩膀/衣服纹理误检为多张脸，IoU 都不高无法用 NMS 合并，但都关联到同一 `person_id`，按 person_id 聚合即可保证"每人 1 个红框"）。

#### track_id 生命周期（遮挡 / 离开场景）

```mermaid
stateDiagram-v2
    [*] --> Tracked: 新目标进入<br/>分配新 track_id
    Tracked --> Tracked: 正常移动 / 短时遮挡<br/>(丢失 ≤ track_buffer 帧, 同号保持)
    Tracked --> Lost: 目标消失 / 完全遮挡 / 匹配失败
    Lost --> Tracked: track_buffer 帧内重现<br/>(恢复原 track_id)
    Lost --> Removed: 丢失超过 track_buffer 帧<br/>轨迹被清理
    Removed --> Tracked: 再次进入 → 分配全新 track_id
    Removed --> [*]
```

> `track_buffer` 单位为**帧**（默认 30）；换算容忍秒数 = `track_buffer ÷ 抽帧fps`，10fps 时约 3 秒。号只增不复用。

---

## 五、VideoTag —— 视频标注模块

### 5.1 输入输出

- **输入**：`101_buffer/device_N/*.bin`、`201_buffer/device_N/*.bin`、原始视频文件
- **输出**：`annotated_buffer/device_N/{起始ts}_{结束ts}.mp4`、`logs/<date>/tag.log`

### 5.2 分层结构

```
VideoTag/
├── api.py               # API 层：Flask HTTP（5003）
├── main.py              # 服务层：CLI 入口
├── manager.py           # 服务层：多设备标注线程管理
├── config_loader.py     # 配置层：读 config.yaml 的 tag 段
├── logger.py            # 基础层：logs/<date>/tag.log
├── protocol_decoder.py  # 数据层：101/201 校验解析，重组 {ts: {persons, faces}}
└── segment_processor.py # 引擎层：分段读取、绘框、编码输出
```

### 5.3 工作流程

1. **解码**：遍历设备协议目录，校验魔数/总长/CRC32，按 ts 重组内存字典；坏文件记 WARN 跳过，不中断。
2. **分段**：按 `segment_seconds`（默认 60）切分。
3. **读帧**：`cv2.VideoCapture(source)`，逐帧 `ts = start_ts + int(frame_idx/fps)`（`start_ts=0` 时取协议最早 ts）。
4. **绘制**：绿框人体 + `ID:track_id`、红框人脸 + `F:face_id`、白框 ROI；颜色/粗细/字号来自配置。
5. **编码**：`cv2.VideoWriter`，先写 `{ts}_{ts}.tmp.mp4`，完成后 `os.replace` 为 `.mp4`，避免半成品。
6. **NVR 模式**：本期未实现，遇到流地址抛 `NotImplementedError` 并记日志（预留扩展点）。

#### 标注时序

```mermaid
sequenceDiagram
    participant S as SegmentProcessor
    participant D as protocol_decoder
    participant V as cv2.VideoCapture
    participant W as cv2.VideoWriter

    S->>D: load_device(101_buffer, 201_buffer, device_N)
    D->>D: 校验魔数/总长/CRC32（坏文件 WARN 跳过）
    D-->>S: {ts: {persons, faces}}
    S->>V: 打开原始视频
    loop 每帧
        V-->>S: frame, frame_idx
        S->>S: ts = start_ts + int(frame_idx/fps)
        alt 跨入新分段
            S->>W: release 旧段 → os.replace .tmp.mp4 为 .mp4
            S->>W: 新建 {seg_start}_{seg_end}.tmp.mp4
        end
        S->>S: 查 ts 注解，绘绿(人体)/红(人脸)/白(ROI)框
        S->>W: write(frame)
    end
    S->>W: 收尾 release → 重命名最后一段
    Note over S,W: 输出 annotated_buffer/device_N/{ts}_{ts}.mp4
```

> **编码器说明**：默认 `fourcc=mp4v`（OpenCV 自带，开箱即用）。若需真正 H.264 请将配置改为 `avc1` 并确保 openh264 可用；不可用时代码会自动回退 `mp4v` 并记 WARN。临时文件保留 `.mp4` 后缀，否则 OpenCV 会按扩展名误选图像序列后端导致初始化失败。

---

## 六、101 / 201 二进制协议规范

统一 **Big-Endian**。整帧结构：

| 字段 | 类型 | 说明 |
|---|---|---|
| start_magic | 4B = `AA 55 AA 55` | 起始魔数 |
| total_length | uint32 | 整帧总字节数（含魔数与 CRC） |
| protocol_no | uint16 | `101` 或 `201` |
| device_no | uint32 | 设备号数字部分 |
| unix_ts | uint64 | Unix 秒级时间戳 |
| record_count | uint16 | 记录条数 |
| body | record_count × 记录 | 见下 |
| crc32 | uint32 | 对 `start_magic..body` 做 `zlib.crc32` |
| end_magic | 4B = `55 AA 55 AA` | 结束魔数 |

固定帧开销 = 4 + 20(帧头) + 4(CRC) + 4 = **32 字节**。

#### 帧字节布局图

```
偏移  字段           大小   说明
┌──────────────────────────────────────────────────────┐
│ 0    start_magic    4B    AA 55 AA 55                  │
│ 4    total_length   4B    uint32  整帧总字节数          │ ┐
│ 8    protocol_no    2B    uint16  101 / 201            │ │
│ 10   device_no      4B    uint32  设备号               │ │ CRC32
│ 14   unix_ts        8B    uint64  Unix 秒               │ │ 覆盖
│ 22   record_count   2B    uint16  记录条数             │ │ 范围
│ 24   body           N×R   record_count 条记录(见下)    │ ┘
│ ...  crc32          4B    uint32  zlib.crc32           │
│ ...  end_magic      4B    55 AA 55 AA                  │
└──────────────────────────────────────────────────────┘
        body 单条记录:
        101 (26B): [target_id 4][x1 4][y1 4][x2 4][y2 4][conf 4][class 2]
        201(2076B): [person_id 4][x1 4][y1 4][x2 4][y2 4][face_id 4][conf 4][embedding 512×4=2048]
```

> 空 body 时整帧 = 32B；101 每条 +26B，201 每条 +2076B。
> 例：1 条人脸 → 32+2076 = **2108B**；2 个人体 → 32+26×2 = **84B**。

**101 记录（人体，26 字节）** `>IiiiifH`：

| target_id u32 | x1 i32 | y1 i32 | x2 i32 | y2 i32 | confidence f32 | class_id u16(=0) |

**201 记录（人脸，2076 字节）** `>IiiiiIf512f`：

| person_id u32（无关联=`0xFFFFFFFF`） | x1 i32 | y1 i32 | x2 i32 | y2 i32 | face_id u32（=person_id） | face_confidence f32 | embedding 512×f32（2048B） |

> `face_id` 字段值 = `person_id`（即绑定的 track_id），用于让红框 ID 跟随人稳定；每人在该秒内只对应 1 条 201 记录（已在 ModelProcessor 按 person_id 聚合）。

- 存储路径：`101_buffer/device_N/{ts}.bin`、`201_buffer/device_N/{ts}.bin`
- 原子写：先写 `{ts}.bin.tmp` → `fsync` → `os.replace`
- 空 body 合法（该秒无目标时为 32 字节空帧）

---

## 七、配置说明（config/config.yaml）

所有可调参数集中于此，无需改代码。关键项：

```yaml
global:
  capture_buffer: "capture_buffer"     # 输入帧目录
  buffer_101: "101_buffer"             # 人体协议输出
  buffer_201: "201_buffer"             # 人脸协议输出
  annotated_buffer: "annotated_buffer" # 标注视频输出
  logs_dir: "logs"
  gpu: true                            # 启用 GPU；无 CUDA 自动降级 CPU

inference:
  fps: 10                              # 抽帧帧率
  yolo_model: "config/models/yolov8s.pt"
  yolo_conf: 0.25                      # 人体置信度阈值
  person_min_pixels: 32                # 人体框最小边长过滤
  edge_margin: 2                       # 边缘贴边框过滤
  person_min_top_margin: 8             # 顶部贴边过滤：< 此值视为"头部被裁掉只露下半身"（避免绿框鬼影）
  face_det_size: [640, 640]            # InsightFace 检测输入尺寸
  face_root: "config"                  # buffalo_l 存于 config/models/buffalo_l
  idle_timeout: 60                     # 空闲秒数，无新帧自动回收
  tracker: "bytetrack.yaml"            # ultralytics 内置 ByteTrack 配置

tag:
  segment_seconds: 60                  # 分段时长
  person_color: [0, 255, 0]            # 人体框 BGR（绿）
  face_color: [0, 0, 255]              # 人脸框 BGR（红）
  roi_color: [255, 255, 255]           # ROI 框 BGR（白）
  roi: []                              # [[x1,y1,x2,y2], ...]
  line_thickness: 2
  font_scale: 0.6
  fourcc: "mp4v"                       # 编码；H.264 用 avc1（需 openh264）
  start_ts: 0                          # 0=用协议最早 ts 作为视频起始基准

system:
  log_retention_days: 7
```

> 路径相对**项目根目录**解析，与运行时所在目录无关。
> 调跟踪遮挡容忍时长：自定义一份 `bytetrack.yaml` 调大 `track_buffer`（单位：帧）。

---

## 八、HTTP API 使用说明

### 8.1 通用约定

- **统一响应体** `{"code": int, "msg": str, "data": any}`；`code=0` 成功，非 0 同时作为 HTTP 状态码（400/404/500）。
- **device_no** 接受 `"device_0"` / `"0"` / 整数 `0`。
- **单设备单任务**：重复 start 同一设备返回错误（msg 含「已在运行」）。
- **stop** 不传 device_no = 停止该服务全部任务。
- 启动均支持 `--host`（默认 `0.0.0.0`）、`--port`。

### 8.2 VideoCapture 采集服务（端口 8000，FastAPI）

`python VideoCapture/main.py server`  （或 `python VideoCapture/app.py`）

| 接口 | 方法 | 入参 | 说明 |
|---|---|---|---|
| `/api/v1/capture/start` | POST | `{source, sampling, jpg_quality, max_seconds, max_frames, max_disk_mb, account, password, device_no, output_dir}` | 启动单路/多路；详见 [3.6 节](#36-关键参数api-请求体) |
| `/api/v1/capture/stop` | POST | `{device_no}` | 停止单设备；省略停全部 |
| `/api/v1/capture/status` | GET | `?device_no=N` | 查询状态；省略查全部 |

```json
// start 响应
{ "code": 0, "msg": "启动完成", "data": { "device_nos": [0, 1], "errors": [] } }
// status 响应
{ "code": 0, "data": [{ "device_no": 0, "status": "running", "frames_captured": 150, "fps": 30.0, "resolution": "640x480", ... }] }
```

> **FastAPI Swagger 文档**：启动服务后浏览器打开 `http://127.0.0.1:8000/docs` 可在线调试所有接口（推荐）。

### 8.3 InsightfaceModel 推理服务（端口 5001，Flask）

`python InsightfaceModel/api.py --host 0.0.0.0 --port 5001`

| 接口 | 方法 | 入参 | 说明 |
|---|---|---|---|
| `/api/v1/inference/start` | POST | `{"fps":10,"device_no":"device_0"}` | 启动；`fps` 可省略（读配置） |
| `/api/v1/inference/stop` | POST | `{"device_no":"device_0"}` | 停止单设备；省略停全部 |
| `/api/v1/inference/status` | GET | `?device_no=device_0` | 查询状态；省略查全部 |

```json
{"code": 0, "msg": "设备 0 推理任务已启动", "data": {"device_no": 0}}
{"code": 0, "msg": "已停止 1 个设备", "data": {"stopped": [0]}}
{"code": 0, "msg": "ok", "data": [{"device_no": 0, "alive": true, "uptime": 12.3, "error": null}]}
{"code": 404, "msg": "设备 0 不存在", "data": null}
```

### 8.4 VideoTag 标注服务（端口 5003，Flask）

`python VideoTag/api.py --host 0.0.0.0 --port 5003`

| 接口 | 方法 | 入参 | 说明 |
|---|---|---|---|
| `/api/v1/tag/start` | POST | `{"source":"./test.mp4","device_no":"device_0"}` | 启动；`source` 必填（缺失返回 400） |
| `/api/v1/tag/stop` | POST | `{"device_no":"device_0"}` | 停止单设备；省略停全部 |
| `/api/v1/tag/status` | GET | `?device_no=device_0` | 查询状态（含已完成分段数 `segments`） |

```json
{"code": 0, "msg": "设备 0 标注任务已启动", "data": {"device_no": 0}}
{"code": 400, "msg": "缺少参数 source", "data": null}
{"code": 0, "msg": "ok", "data": [{"device_no": 0, "alive": false, "segments": 5, "uptime": 30.2, "error": null}]}
```

### 8.5 curl 示例

Linux / macOS：

```bash
# VideoCapture
curl -X POST http://localhost:8000/api/v1/capture/start \
  -H "Content-Type: application/json" \
  -d '{"source":"demo_videos/video1.mp4","sampling":1,"jpg_quality":75}'

# InsightfaceModel
curl -X POST http://localhost:5001/api/v1/inference/start \
  -H "Content-Type: application/json" \
  -d '{"fps": 10, "device_no": "device_0"}'

# VideoTag
curl -X POST http://localhost:5003/api/v1/tag/start \
  -H "Content-Type: application/json" \
  -d '{"source": "./test_video.mp4", "device_no": "device_0"}'
```

Windows（cmd 需转义双引号，不能用 `\` 续行）：

```bat
curl -X POST http://localhost:8000/api/v1/capture/start -H "Content-Type: application/json" -d "{\"source\":\"demo_videos/video1.mp4\",\"sampling\":1,\"jpg_quality\":75}"
curl -X POST http://localhost:5001/api/v1/inference/start -H "Content-Type: application/json" -d "{\"fps\":10,\"device_no\":\"device_0\"}"
curl -X POST http://localhost:5003/api/v1/tag/start -H "Content-Type: application/json" -d "{\"source\":\"./test_video.mp4\",\"device_no\":\"device_0\"}"
```

---

## 九、核心设计规范

1. **目录解耦**：模块间仅读写约定缓存目录，无代码导入依赖，支持单模块离线复现。
2. **配置驱动**：阈值、路径、时长、视觉参数全部外置 YAML，硬编码最小化。
3. **原子写入**：协议 `.bin`、视频 `.mp4` 均先写临时文件再 rename，下游不读半成品。
4. **资源自动释放**：模型/GPU/句柄在空闲回收或任务停止时释放。
5. **时间戳统一**：全系统以 Unix 秒级时间戳为唯一主键。
6. **异常可恢复**：模型加载、协议校验、视频读写、编码失败均捕获记日志，不崩溃进程。
7. **日志标准化**：标准库 logging，按日期输出，分级 INFO/WARN/ERROR；VideoCapture 日志中 RTSP 密码自动脱敏。

---

## 十、测试与验收

- **协议往返**：`encode_101/201` → `protocol_decoder` 解析，断言魔数/CRC/字段（含 512 维浮点）完全还原；坏 CRC 被拒。
- **配置校验**：字段缺失 / 路径错误 / 阈值越界被 `config_loader` 捕获报错。
- **VideoCapture 端到端**：本地文件 / USB 摄像头 / RTSP 三类源都能稳定落盘到 `capture_buffer/device_N/<ts>/*.jpg`；`max_frames` / `max_disk_mb` 超限自动删最旧；RTSP 断线自动重连；RTSP 密码不在日志明文出现。
- **InsightfaceModel 端到端**：给定 `capture_buffer` 帧，持续生成合法 101/201、无 `.tmp` 残留、空闲自动回收，CPU/GPU 均可。
- **VideoTag 端到端**：协议 + 测试视频 → 可播放 MP4、框位置匹配、分段命名正确；删一个 bin 验证缺协议记 WARN 不中断。
- **全链路**：VideoCapture → InsightfaceModel → VideoTag 串联，目标时序与 ID 与协议一致。

---

## 十一、开源参考

- 人体检测：[ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)
- 人脸特征：[deepinsight/insightface](https://github.com/deepinsight/insightface)（buffalo_l，512 维）
- 多目标跟踪：ByteTrack（ultralytics 内置）
- 视频读写/绘制：OpenCV `VideoCapture` / `VideoWriter`
