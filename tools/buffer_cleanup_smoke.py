"""Verify occupied files retry without losing published videos."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from VideoTag.segment_processor import SegmentProcessor
from buffer_cleanup import BufferCleaner
import json

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    tag = SegmentProcessor.__new__(SegmentProcessor)
    tag.device_no = 1
    tag.capture_dir = root / 'raw'
    tag.buffer_101 = root / '101'
    tag.buffer_201 = root / '201'
    tag.cleanup_raw_frames = True
    tag.cleanup_protocol = True
    tag._cleanup_pending = set()
    raw = tag.capture_dir / '100'
    raw.mkdir(parents=True)
    (raw / 'frame.jpg').write_bytes(b'raw')
    for folder in (tag.buffer_101, tag.buffer_201):
        (folder / 'device_1').mkdir(parents=True)
        (folder / 'device_1/100.bin').write_bytes(b'inference')
    sidecar = tag.buffer_101 / 'device_1/100.frames.json'
    sidecar.write_text('{}')
    video = root / '100.mp4'
    video.write_bytes(b'published video')
    with patch('VideoTag.segment_processor.shutil.rmtree', side_effect=PermissionError('busy')):
        tag._cleanup_second(100)
    assert tag._cleanup_pending == {100}
    assert raw.exists() and not sidecar.exists()
    tag._retry_cleanup()
    assert not raw.exists() and not tag._cleanup_pending
    assert video.read_bytes() == b'published video'
    raw.mkdir()
    tag.cleanup_raw_frames = tag.cleanup_protocol = False
    tag._cleanup_second(100)
    assert raw.exists()
    state = root / 'state'
    state.mkdir()
    config = {'global': {'capture_buffer': str(tag.capture_dir),
                         'buffer_101': str(tag.buffer_101), 'buffer_201': str(tag.buffer_201)}}
    capture = tag.capture_dir / 'device_1'
    for ts in (50, 100, 101):
        (capture / str(ts)).mkdir(parents=True)
        (capture / str(ts) / 'frame.jpg').write_bytes(b'frame')
        for folder in (tag.buffer_101, tag.buffer_201):
            (folder / f'device_1/{ts}.bin').write_bytes(b'inference')
    other = tag.buffer_101 / 'device_2'
    other.mkdir()
    (other / '50.bin').write_bytes(b'other device')
    cleaner = BufferCleaner(config, 1, state, 100)
    cleaner.sweep()
    assert not (capture / '50').exists()
    assert (capture / '100').exists() and (capture / '101').exists()
    (state / 'tag.metrics.json').write_text(json.dumps({'last_ts':100}))
    cleaner.sweep()
    assert not (capture / '100').exists() and (capture / '101').exists()
    assert (other / '50.bin').exists() and video.exists()
print('PASS: runtime cleanup, busy-file retry, preserved output and test-mode retention')
