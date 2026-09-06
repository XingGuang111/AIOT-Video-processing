from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

class CaptureStatusEnum(str, Enum):
    INIT = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

@dataclass
class CaptureStatus:
    device_no: Optional[int] = None
    source: str = ""
    source_type: str = ""
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    frame_count: int = 0
    frames_captured: int = 0
    status: CaptureStatusEnum = CaptureStatusEnum.INIT
    last_error: str = ""
    start_time: float = 0.0

    def to_dict(self):
        data = asdict(self)
        data["resolution"] = f"{self.resolution[0]}x{self.resolution[1]}"
        return data
