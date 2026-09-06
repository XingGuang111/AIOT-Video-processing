# VideoCapture —— 视频帧采集模块

## 项目概述

VideoCapture 是视频数据采集与帧数据落盘模块，位于边缘计算系统或视频分析系统的数据输入端。模块从 RTSP 摄像头、网络视频流或本地视频文件中连续读取画面，将每一帧编码为 JPEG 图片，并按照"设备编号/秒级时间戳/帧序号"的目录结构保存到本地缓冲区，供下游算法模块读取。

模块不直接承担目标检测、识别、告警等业务逻辑，只负责稳定、连续、可追溯地生成帧图片数据。该设计降低采集模块与算法模块的耦合度，便于多种算法以统一方式读取视频帧。

**系统调用链路：** `[视频源] → VideoCapture → output_buffer → 下游模块读取 → 算法处理/入库/展示`

---

多路视频帧采集服务，支持本地视频文件、RTSP 流、USB 摄像头等多种视频源，按秒级时间戳目录输出 JPEG 帧。模块不直接承担目标检测、识别、告警等业务逻辑，只负责稳定、连续、可追溯地生成帧图片数据，供下游算法模块统一读取。## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                      app.py (FastAPI)                    │
│  POST /start   POST /stop   GET /status                 │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│                  capture_manager.py                      │
│  设备编号分配  启停控制  状态查询  线程管理                 │
└────────────┬────────────────────────────────────────────┘
             │ 1:N
┌────────────▼──────┐  ┌────────────▼──────┐
│  capture_worker   │  │  capture_worker   │  ...
│  (Thread-0)       │  │  (Thread-1)       │
│  ┌──────────────┐ │  │  ┌──────────────┐ │
│  │ source_parser │ │  │  │ source_parser │ │
│  │   识别类型     │ │  │  │   识别类型     │ │
│  ├──────────────┤ │  │  ├──────────────┤ │
│  │ cv2.VideoCap │ │  │  │ cv2.VideoCap │ │
│  │ 读取→编码→写盘 │ │  │  │ 读取→编码→写盘 │ │
│  ├──────────────┤ │  │  ├──────────────┤ │
│  │ _enforce_*   │ │  │  │ _enforce_*   │ │
│  │ 缓存限制检查   │ │  │  │ 缓存限制检查   │ │
│  └──────────────┘ │  │  └──────────────┘ │
└──────┬────────────┘  └──────┬────────────┘
       │ device_0/            │ device_1/
       ▼                      ▼
┌──────────────────────────────────────┐
│           BufferCleaner              │
│  定时清理  max_seconds / max_frames  │
│           max_disk_mb / total_...    │
└──────────────────────────────────────┘
```


## 1. 目录结构

```
VideoCapture/
├── app.py                         # API 服务入口（FastAPI 路由层）
├── main.py                        # 命令行入口
├── config.yaml                    # 全局配置文件
├── requirements.txt               # Python 依赖
├── README.md
├── videocapture/                  # 核心包
│   ├── __init__.py                # 统一导出 + start_capture 便捷函数
│   ├── capture_manager.py         # 多路采集管理器
│   ├── capture_worker.py          # 单路采集线程
│   ├── buffer_cleaner.py          # 缓冲区容量控制
│   ├── source_parser.py           # 视频源类型识别与 URL 构造
│   ├── models.py                  # 状态枚举 & 数据模型
│   └── logger.py                  # 日志配置
├── demo_videos/                   # 示例视频
├── extracted_frames/              # 帧输出目录（默认）
│   └── device_0/
│       └── 1717891200/            # Unix 时间戳秒目录
│           ├── frame_000000.jpg
│           └── frame_000001.jpg
└── logs/                          # 运行日志
```

## 2. 模块职责

| 模块 | 职责 |
|------|------|
| `app.py` | 仅定义 FastAPI 接口，业务逻辑委托给 `CaptureManager` |
| `main.py` | 命令行入口，支持 `server` / `extract` 子命令 |
| `capture_manager.py` | 管理多路 worker 创建、启停、状态查询，线程安全，自动分配设备编号 |
| `capture_worker.py` | 单路采集线程：视频源初始化、帧读取、限速、重连、帧数/磁盘限制 |
| `buffer_cleaner.py` | 后台线程，按时间/帧数/磁盘容量自动清理旧帧 |
| `source_parser.py` | 识别 file/rtsp/device/stream 类型，纯 IP 自动拼接海康 RTSP URL |
| `models.py` | `CaptureStatusEnum` 状态枚举 + `CaptureStatus` 数据类 |
| `logger.py` | 统一日志初始化（文件+控制台），日志中自动脱敏 RTSP 密码 |

---

## 3. 快速开始

### 3.1 环境要求

- Python >= 3.10
- 操作系统：Windows / Linux / macOS

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 启动 API 服务

```bash
python main.py server
```

服务启动后访问接口文档：http://127.0.0.1:8000/docs

### 3.4 命令行直接提取

```bash
# 提取本地视频文件
python main.py extract demo_videos/video1.mp4

# 提取 RTSP 流
python main.py extract rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101

# 提取 USB 摄像头（设备号 0）
python main.py extract 0
```

按 `Ctrl+C` 停止采集。

---

## 4. 入口函数

### 4.1 函数签名

```python
from videocapture import start_capture

def start_capture(
    source: str,
    account: str | None = None,
    password: str | None = None,
    device_no: int | None = None,
    output_dir: str = "extracted_frames",
    jpg_quality: int = 75,
    max_seconds: int | None = None,
    max_frames: int | None = None,
    max_disk_mb: int | None = None,
    stop_event: threading.Event | None = None,
) -> int:
```

### 4.2 参数详解

| 参数 | 类型 | 必填 | 默认值 | 实现机制 |
|------|------|------|--------|----------|
| `source` | str | 是 | — | 经 `parse_source()` 自动识别 file/rtsp/device/stream，传给 `cv2.VideoCapture()` |
| `account` | str | 否 | None | 纯 IP 时拼接 RTSP：`rtsp://account:password@ip:554/...` |
| `password` | str | 否 | None | 同上，日志中 `desensitize_rtsp_url()` 脱敏为 `***` |
| `device_no` | int | 否 | None | 为空时 `CaptureManager._next_device_no` 自增分配，控制输出目录 `device_N` |
| `output_dir` | str | 否 | extracted_frames | `Path(output_dir) / device_N` 创建输出根目录 |
| `jpg_quality` | int | 否 | 75 | `cv2.imencode()` JPEG 编码质量参数 |
| `max_seconds` | int | 否 | None | 实时流每帧检查 `elapsed >= max_seconds`，触发自动停止 |
| `max_frames` | int | 否 | None | 每 30 帧检查，超出则 `shutil.rmtree` 删除最旧秒目录 |
| `max_disk_mb` | int | 否 | None | 每 30 帧检查目录字节数，超出则删除最旧秒目录 |
| `stop_event` | Event | 否 | None | 外部停止信号，不传则 worker 内部创建；`stop()` set 后 `run()` 循环退出 |

### 4.3 调用示例

```python
from videocapture import start_capture

# 本地视频文件
d0 = start_capture("demo_videos/video1.mp4")

# USB 摄像头，30 秒自动停止，JPEG 质量 75
d1 = start_capture("0", jpg_quality=75, max_seconds=30)

# RTSP 摄像头，最多 500 帧，不超过 50MB
d2 = start_capture("192.168.1.64", account="admin", password="12345",
                   max_frames=500, max_disk_mb=50)

# 外部停止信号控制
import threading
evt = threading.Event()
d3 = start_capture("0", stop_event=evt)
# ... 业务逻辑 ...
evt.set()  # 通知停止
```

---

## 5. API 接口

### 5.1 启动采集（单路/多路）

```
POST /api/v1/capture/start
```

**单路请求体：**

```json
{ "source": "demo_videos/video1.mp4", "jpg_quality": 75, "max_seconds": 0 }
```

**多路请求体：**

```json
{ "source": ["demo_videos/video1.mp4", "0"], "jpg_quality": 75, "max_seconds": 30 }
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `source` | string / string[] | 是 | — | 字符串为单路，列表为多路 |
| `account` | string | 否 | null | RTSP 账号（纯 IP 时必填） |
| `password` | string | 否 | null | RTSP 密码 |
| `device_no` | int | 否 | null | 设备编号，为空自动分配 |
| `output_dir` | string | 否 | config 配置 | 输出根目录 |
| `jpg_quality` | int | 否 | 75 | JPEG 质量 1-100 |
| `sampling` | float | 否 | 1.0 | 采样间隔（秒），-1=全量提取 |
| `max_seconds` | int | 否 | null | 最大采集时长（秒），null=不限制 |
| `max_frames` | int | 否 | null | 最大帧数限制，超限删最旧 |
| `max_disk_mb` | int | 否 | null | 最大磁盘占用（MB），超限删最旧 |

**响应：**

```json
{ "code": 0, "msg": "启动完成", "data": { "device_nos": [0, 1], "errors": [] } }
```

### 5.2 停止采集

```
POST /api/v1/capture/stop
```

```json
{ "device_no": 0 }
```

`device_no` 为空则停止全部设备。

### 5.3 查询状态

```
GET /api/v1/capture/status?device_no=0
```

`device_no` 为空则返回全部设备。

**响应：**

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

**状态枚举：**

| 状态 | 说明 |
|------|------|
| `initialized` | 已初始化 |
| `running` | 采集中 |
| `finished` | 文件播放完毕 / 达到 max_seconds 自动停止 |
| `error` | 异常 |
| `reconnecting` | RTSP 重连中 |
| `stopped` | 已停止 |

---

## 6. 视频源类型识别

### 6.1 识别原理

`source_parser.py` 采用**字符串前缀匹配**算法，不进行网络探测。按以下优先级依次判定：

```
输入 source
  ├─ 纯数字？         → device (USB 摄像头)
  ├─ "rtsp://" 开头？ → rtsp   (RTSP 实时流)
  ├─ 纯 IPv4 地址？   → rtsp   (拼接海康默认 RTSP URL)
  ├─ "http(s)://" 开头？
  │   ├─ URL 路径以 .mp4/.avi 等结尾？ → file   (网络视频文件)
  │   └─ 否则                         → stream (网络视频流)
  └─ 以上都不满足     → file   (本地/未知文件)
```

**纯 IPv4 判断**：按 `.` 拆分为 4 段，每段检查 `0 <= n <= 255`。

**视频扩展名检测**：通过 `urllib.parse.urlparse()` 提取 URL 路径，匹配预定义扩展名集合 `{.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v, .ts}`。

### 6.2 识别规则表

| 输入示例 | 识别类型 | 判定路径 | 处理方式 |
|----------|----------|----------|----------|
| `0`, `1` | `device` | `isdigit()` | `cv2.VideoCapture(设备号)` |
| `rtsp://admin:pwd@ip:554/...` | `rtsp` | `startswith("rtsp://")` | 直接作为 VideoCapture 输入 |
| `192.168.1.64` | `rtsp` | `is_ipv4()` | 拼接 `rtsp://account:password@ip:554/Streaming/Channels/101` |
| `demo/video1.mp4` | `file` | 兜底匹配 | 直接打开文件路径 |
| `http://server/video.mp4` | `file` | HTTP + 视频扩展名 | 按文件 URL 打开，视为文件模式 |
| `http://ip/live/stream` | `stream` | HTTP + 非视频扩展名 | 按网络流处理，失败进行重试 |

---

## 7. 配置文件

`config.yaml`：

```yaml
# 全局配置
global:
  output_dir: "extracted_frames"   # 输出根目录
  default_fps: 25                   # 默认帧率（视频源 FPS 不可用时回退）

# RTSP 重连配置
reconnect:
  max_reconnect_times: 10           # 最大重连次数
  reconnect_interval: 1             # 重连间隔（秒）
  max_frame_fails: 5                # 连续读帧失败阈值（触发重连）

# 缓存清理配置
buffer_clean:
  max_seconds: 600                  # 保留最近 N 秒的帧（0=不限制）
  max_frames: 0                     # 最多保留 N 帧（0=不限制）
  max_disk_mb: 2048                 # 单路最多占用 N MB 磁盘（0=不限制）
  total_max_disk_mb: 10240           # 全局最多占用 N MB 磁盘（0=不限制）
  clean_interval: 15                # 清理检查间隔（秒）
```

---

## 8. 输出目录结构

帧文件按 **设备号 → Unix 时间戳（秒）→ 帧序号** 三级目录组织：

```
extracted_frames/
├── device_0/          ← 第 1 路视频源
│   ├── 1717891200/
│   │   ├── frame_000000.jpg
│   │   └── frame_000001.jpg
│   └── 1717891201/
│       └── frame_000000.jpg
└── device_1/          ← 第 2 路视频源
    └── 1717891200/
        └── frame_000000.jpg
```

- **实时流（RTSP/摄像头）**：时间戳为当前系统时间
- **文件模式**：时间戳基于视频播放进度推算，并按原始 FPS 限速以模拟真实播放

---

## 9. 核心设计要点

### 9.1 文件模式限速策略

视频文件读取速度通常高于真实播放速度。文件模式全量提取（`sampling=-1`）时，按视频原始 FPS 进行限速：

```
expected_elapsed = frame_index / fps
actual_elapsed = time.time() - start_real_time
sleep_time = expected_elapsed - actual_elapsed
```

### 9.2 RTSP 断线重连

- 单次读帧失败不立即退出，等待重试
- 连续失败达到 `max_frame_fails` 阈值后触发重连
- 重连超过 `max_reconnect_times` 后标记为 `error` 状态
- 停止信号优先级高于重连逻辑

### 9.3 缓存控制（三级限制）

```
采集循环（每 30 帧检查）              BufferCleaner 后台线程（每 N 秒）
┌─────────────────────────┐          ┌──────────────────────────┐
│ max_seconds              │          │ max_seconds              │
│   → 达到后自动 finished   │          │   → 删除早于 now-N 的目录 │
│ max_frames               │          │ max_frames               │
│   → 超出删本路最旧秒目录   │          │   → 全局统计超出删最旧     │
│ max_disk_mb              │          │ max_disk_mb              │
│   → 超出删本路最旧秒目录   │          │   → 全局统计超出删最旧     │
└─────────────────────────┘          │ total_max_disk_mb         │
                                      │   → 全局上限超出删最旧     │
                                      └──────────────────────────┘
```

- **单路限制**：`max_frames`、`max_disk_mb` 作用域为当前 device_N 目录，在采集循环中实时检查
- **全局限制**：`total_max_disk_mb` 作用域为全部 device_N 目录，由 BufferCleaner 定时检查
- **淘汰策略**：一律按秒目录 Unix 时间戳升序排列，优先删除最旧数据

### 9.4 日志安全

日志中自动对 RTSP 地址脱敏，密码字段替换为 `***`，不会明文输出摄像头密码。

---

## 10. 开发建议

推荐按以下顺序逐步实现和验证：

1. 本地视频文件抽帧，完成 JPEG 输出和目录结构创建
2. RTSP 地址打开与摄像头实时流采集
3. `source_parser.py` 统一源类型识别和 RTSP URL 构造
4. `CaptureWorker` 封装单路采集线程
5. `CaptureManager` 管理多路视频源、设备编号和停止信号
6. `BufferCleaner` 定期清理超时/超帧数/超磁盘空间的数据
7. 封装 start / stop / status API，供上层系统调用
8. 补充日志、异常处理和单元测试

---

## 11. 依赖环境

| 包名 | 版本要求 | 说明 |
|------|----------|------|
| opencv-python | >= 4.5 | 视频读取与帧编码 |
| fastapi | >= 0.100 | API 服务框架 |
| uvicorn | >= 0.23 | ASGI 服务器 |
| pyyaml | >= 6.0 | 配置文件解析 |
| pydantic | >= 2.0 | 请求体校验 |
| python-multipart | >= 0.0.6 | 文件上传支持 |

## 12. 操作示例

先启动服务：`python main.py server`，接口：http://127.0.0.1:8000/api/v1/capture/start

### 12.1 本地视频逐帧转照片（全量提取）

```python
{ "source": "demo_videos/video3.mp4", "sampling": -1, "jpg_quality": 75 }
```

### 12.2 本地视频按秒抽帧

```python
{ "source": "VideoCapture/demo_videos/video1.mp4", "sampling": 1, "jpg_quality": 75 }
```

### 12.3 两个视频同时转

```python
{ "source": ["demo_videos/video1.mp4", "demo_videos/video2.mp4"], "sampling": 1, "jpg_quality": 75 }
```

输出各自独立：`device_0/` ← video1，`device_1/` ← video2

### 12.4 只转摄像头

```python
{ "source": "0", "jpg_quality": 75, "max_seconds": 60, "max_frames": 500, "max_disk_mb": 50 }
```

| 参数        | 值   | 含义                             |
| ----------- | ---- | -------------------------------- |
| source      | "0"  | 第一个摄像头（多个用 "1"、"2"）  |
| max_seconds | 60   | 60秒自动停（0=不限时，Ctrl+C停） |
| max_frames  | 500  | 超500帧删最旧                    |
| max_disk_mb | 50   | 超50MB删最旧                     |

### 12.5 摄像头+本地视频一起

```
{ "source": ["demo_videos/video1.mp4", "0"], "sampling": 1, "jpg_quality": 75, "max_seconds": 300 }
```

### 12.6 RTSP 摄像头

已知地址：

```
{ "source": "rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101", "jpg_quality": 75, "max_seconds": 300 }
```

纯IP需账号密码：

```
{ "source": "192.168.1.64", "account": "admin", "password": "12345", "jpg_quality": 75, "max_seconds": 300 }
```

### 12.7 汇总对照

| 场景        | source                | 关键参数             |
| ----------- | --------------------- | -------------------- |
| 本地逐帧    | `"video1.mp4"`        | `sampling=-1`        |
| 本地按秒    | `"video1.mp4"`        | `sampling=1`         |
| 双视频并发  | `["v1.mp4","v2.mp4"]` | —                    |
| 摄像头限时  | `"0"`                 | `max_seconds=60`     |
| 摄像头不限  | `"0"`                 | `max_seconds=0`      |
| 视频+摄像头 | `["v1.mp4","0"]`      | `max_seconds=300`    |
| RTSP 地址   | `"rtsp://..."`        | `max_seconds=300`    |
| RTSP IP     | `"192.168.1.64"`      | `account`+`password` |