"""Concatenate already encoded, matching segments without decoding/re-encoding."""
import os
from pathlib import Path
import subprocess
import time
import uuid

import cv2


def concatenate_segments(paths, destination, cancelled=lambda: False):
    import imageio_ffmpeg
    paths = [Path(path) for path in paths]
    destination = Path(destination)
    token = uuid.uuid4().hex
    manifest = destination.parent / f'.concat_{token}.txt'
    error_log = destination.parent / f'.concat_{token}.log'
    temporary = destination.with_name(destination.stem + '.tmp.mp4')
    expected = 0
    reference = None
    for path in paths:
        if cancelled():
            raise InterruptedError('停止合并视频')
        cap = cv2.VideoCapture(str(path))
        try:
            info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count < 1:
                raise IOError(f'片段无法播放: {path}')
            if reference and (info[:2] != reference[:2] or abs(info[2] - reference[2]) > 0.02):
                raise IOError('片段分辨率或帧率不一致')
            reference = info
            expected += count
        finally:
            cap.release()
    if not expected:
        raise IOError('没有可合并片段')
    child = None
    try:
        # Relative manifest entries avoid shell escaping and work with Unicode directories.
        manifest.write_text(''.join("file '" + os.path.relpath(p, destination.parent).replace('\\', '/') + "'\n"
                                    for p in paths), encoding='utf-8')
        with error_log.open('wb') as errors:
            child = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                                      '-nostdin', '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                                      '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart', str(temporary)],
                                     stdout=subprocess.DEVNULL, stderr=errors,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            while child.poll() is None:
                if cancelled():
                    raise InterruptedError('停止合并视频')
                time.sleep(0.05)
        if child.returncode:
            raise IOError('视频无损合并失败: ' + error_log.read_text(encoding='utf-8', errors='replace')[-2000:])
        cap = cv2.VideoCapture(str(temporary))
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected or not cap.read()[0]:
                raise IOError('完整视频帧数校验失败')
        finally:
            cap.release()
        os.replace(temporary, destination)
        return expected
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for path in (manifest, error_log, temporary):
            path.unlink(missing_ok=True)
