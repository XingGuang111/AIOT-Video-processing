"""Small per-process timing records, published atomically for the local dashboard."""
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.03 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


class Metrics:
    def __init__(self, stage):
        directory = os.environ.get('AIOT_PIPELINE_STATE')
        self.path = Path(directory) / (stage + '.metrics.json') if directory else None
        self.data = {'started_at': time.time(), 'timings': {}, 'processed': 0, 'frames': 0}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            timing = self.data['timings'].setdefault(name, {'seconds': 0, 'calls': 0})
            timing['seconds'] += time.perf_counter() - started
            timing['calls'] += 1

    def publish(self, finished=False):
        self.data['updated_at'] = time.time()
        if finished:
            self.data['ended_at'] = self.data['updated_at']
        if self.path:
            write_json(self.path, self.data)
