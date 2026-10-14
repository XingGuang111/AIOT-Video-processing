# -*- coding: utf-8 -*-
"""Verify sequential MJPEG frames, pacing, structured logs and record cleanup."""
import json
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from InsightfaceModel.config_loader import load_config
from webapp.server import create_app


def main():
    work = Path(tempfile.mkdtemp(prefix='playback_contract_', dir=ROOT / 'logs'))
    config = load_config()
    for key in ('capture_buffer', 'buffer_101', 'buffer_201', 'annotated_buffer', 'logs_dir'):
        config['global'][key] = str(work / key)
    app = create_app(config)
    service = app.state.service
    identifier = 'a' * 32
    job = dict(id=identifier, device_no=7, created_at=100, ended_at=110,
               source='test', requested_stop=False, exit_code=0, pid=0)
    service._jobs[identifier] = job
    service._save(job)
    state = service.state_dir(job)
    state.mkdir()
    (state / 'run.json').write_text(json.dumps({'start_ts': 100, 'file_realtime':False}))
    (state / 'capture.done').write_text(json.dumps({'seconds':3,'frames':15}))
    directory = work / 'annotated_buffer/device_7/segments'
    directory.mkdir(parents=True)
    for ts in (100, 101, 102, 103):
        writer = cv2.VideoWriter(str(directory / f'{ts}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (160, 120))
        for frame in range(5):
            writer.write(np.full((120, 160, 3), ts + frame, np.uint8))
        writer.release()
    (service.job_root / f'{identifier}.log').write_text(
        '[capture] [2026-10-13 12:00:00,000] [INFO] 秒目录完成 ts=100 帧=5\n'
        '[inference] [2026-10-13 12:00:01,000] [INFO] ts=100 人体=2 人脸=1\n'
        '[tag] [2026-10-13 12:00:02,000] [INFO] 秒段完成 ts=100 帧=5\n', encoding='utf-8')
    with TestClient(app) as client:
        started = time.monotonic()
        response = client.get(f'/api/stream/7?start_ts=100&job_id={identifier}')
        elapsed = time.monotonic() - started
        import re
        timestamps = [int(value) for value in re.findall(rb'X-Segment-Timestamp: (\d+)', response.content)]
        assert timestamps == [100]*5 + [101]*5 + [102]*5, timestamps
        assert elapsed >= 3, elapsed
        events = client.get(f'/api/tasks/{identifier}/logs').json()['events']
        assert [event['stage'] for event in events] == ['capture', 'inference', 'tag']
        assert all(event['second'] == 1 for event in events)
        result = client.get(f'/api/results/7?job_id={identifier}').json()
        assert result['total'] == 3
        assert {item['ts'] for item in result['items']} == {100,101,102}
        assert client.delete('/api/records', headers={'Origin':'https://example.com'}).status_code == 403
        result = client.delete('/api/records').json()
        assert result['deleted'] == 1
        assert not list(service.job_root.glob('*.json'))
        assert not state.exists()
        assert len(list(directory.glob('*.mp4'))) == 4
        assert client.get('/api/overview').json()['jobs'] == []
    print(json.dumps({'passed':['all_frames_in_order','paced_playback','structured_events','clear_records_preserves_video'], 'elapsed':elapsed}))


if __name__ == '__main__':
    main()
