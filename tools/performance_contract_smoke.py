# -*- coding: utf-8 -*-
"""Verify all-frame moving boxes, identity isolation, lossless merge and cancellation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from InsightfaceModel.config_loader import load_config
from InsightfaceModel.model_processor import ModelProcessor
from InsightfaceModel.tracker import Tracker
from VideoTag.frame_annotations import FrameAnnotations
from VideoTag.segment_processor import SegmentProcessor
from video_output import concatenate_segments
from capture_buffers import ready_seconds, buffer_usage


def main():
    work = Path(tempfile.mkdtemp(prefix='performance_contract_', dir=ROOT / 'logs'))
    config = load_config()
    for key in ('capture_buffer', 'buffer_101', 'buffer_201', 'annotated_buffer', 'logs_dir'):
        config['global'][key] = str(work / key)
    source = work / 'capture_buffer/device_5/100'
    source.mkdir(parents=True)
    for index in range(18):
        ok, data = cv2.imencode('.jpg', np.full((120, 160, 3), index + 10, np.uint8))
        assert ok
        data.tofile(str(source / f'frame_{index:06d}.jpg'))
    (source / '_ready.json').write_text(json.dumps({'fps':18, 'frames':18}))
    class Detector:
        def detect(self, frame):
            index = int(frame[60, 80, 0]) - 10
            return [dict(track_id=42, xyxy=(10+index, 15, 45+index, 75), conf=0.9)]
    class Faces:
        def embed(self, frame):
            index = int(frame[60, 80, 0]) - 10
            feature = np.zeros(512, np.float32)
            feature[0] = 1
            return [dict(xyxy=(14+index, 22, 28+index, 35), conf=0.9, embedding=feature)]
    model = ModelProcessor(5, config)
    model.detector, model.embedder, model.tracker = Detector(), Faces(), Tracker()
    model._process_second(100)
    path = model.buffer_101 / '100.frames.json'
    samples = json.loads(path.read_text(encoding='utf-8'))['frames']
    assert len(samples) == 10
    assert model.metrics.data['timings']['yolo']['calls'] == 10
    assert model.metrics.data['timings']['face']['calls'] == 10
    interpolated = FrameAnnotations(samples)
    for index in range(18):
        records = interpolated.at(index, {})
        assert records['persons'][0]['xyxy'][0] == 10 + index
        assert records['faces'][0]['xyxy'][0] == 14 + index
        assert records['persons'][0]['track_id'] == 1
        assert records['faces'][0]['face_id'] == 0
    tag = SegmentProcessor(5, '', config, follow=True)
    assert tag._annotate_second(100, threading.Event())
    segment = tag.segments_dir / '100.mp4'
    cap = cv2.VideoCapture(str(segment))
    for index in range(18):
        ok, frame = cap.read()
        assert ok
        green = frame[55, 10+index].astype(int)
        red = frame[27, 14+index].astype(int)
        assert green[1] > green[0] + 60 and green[1] > green[2] + 60, (index, green)
        assert red[2] > red[0] + 60 and red[2] > red[1] + 60, (index, red)
    assert not cap.read()[0]
    cap.release()
    assert path.exists()
    full = tag.out_dir / '100_100_full.mp4'
    assert concatenate_segments([segment], full) == 18
    cancelled = tag.out_dir / 'cancelled.mp4'
    try:
        concatenate_segments([segment], cancelled, cancelled=lambda: True)
    except InterruptedError:
        pass
    else:
        raise AssertionError('cancel ignored')
    assert not cancelled.exists()
    # Different identities must never be interpolated into one another.
    timeline = FrameAnnotations([{'index':0,'persons':[{'track_id':1,'xyxy':[0,0,10,10]}]},
                                 {'index':10,'persons':[{'track_id':2,'xyxy':[100,0,110,10]}]}])
    assert timeline.at(5,{})['persons'][0]['xyxy'] == [0,0,10,10]
    # Windows reports PermissionError while a consumed directory is being removed.
    class Entry:
        name = '100'
        path = str(source)
        def is_dir(self, **kwargs):
            raise PermissionError('directory is being removed')
    class Entries:
        def __enter__(self):
            return iter([Entry()])
        def __exit__(self, *args):
            return False
    with patch('capture_buffers.os.scandir', return_value=Entries()):
        assert ready_seconds(source.parent) == []
        assert buffer_usage(source.parent) == (0, 0)
    print(json.dumps({'passed':['original_detection_frequency','18_moving_green_and_red_frames',
                               'no_identity_mix','lossless_merge_frame_count','merge_cancel','sidecar_retained'],
                      'work':str(work)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
