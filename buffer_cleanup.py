"""Delete only device-local intermediate buffers below a consumed watermark."""
import json
import shutil
import threading
from pathlib import Path


class BufferCleaner(threading.Thread):
    def __init__(self, config, device_no, state, start_ts):
        super().__init__(daemon=True)
        self.roots = [Path(config['global'][key]) / f'device_{device_no}'
                      for key in ('capture_buffer', 'buffer_101', 'buffer_201')]
        self.state = Path(state)
        self.watermark = start_ts - 1
        self.stop_event = threading.Event()
        self.deleted = 0

    def sweep(self):
        try:
            metrics = json.loads((self.state / 'tag.metrics.json').read_text(encoding='utf-8'))
            self.watermark = max(self.watermark, int(metrics.get('last_ts', self.watermark)))
        except (OSError, ValueError, TypeError):
            pass
        removed = 0
        for root in self.roots:
            try:
                entries = list(root.iterdir())
            except OSError:
                continue
            for entry in entries:
                stamp = entry.name.split('.', 1)[0]
                if not stamp.isdigit() or int(stamp) > self.watermark:
                    continue
                if entry.is_symlink() or entry.resolve().parent != root.resolve():
                    continue
                # Numeric capture directories, protocol/sidecar and atomic-write leftovers.
                suffix = entry.name[len(stamp):]
                if entry.is_dir() and suffix == '':
                    action = shutil.rmtree
                elif suffix in ('.bin', '.bin.tmp', '.frames.json', '.frames.json.tmp'):
                    action = Path.unlink
                else:
                    continue
                try:
                    action(entry)
                    removed += 1
                except OSError:
                    continue  # Retry next tick while Windows still holds the file.
        self.deleted += removed
        if removed:
            print(f'[pipeline] buffer_cleanup removed={removed} total={self.deleted} through_ts={self.watermark}', flush=True)
        return removed

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sweep()
            except OSError as exc:
                print(f'[pipeline] buffer_cleanup retry: {exc}', flush=True)
            self.stop_event.wait(0.25)

    def close(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout=5)
        self.sweep()
