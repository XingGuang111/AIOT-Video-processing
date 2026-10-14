"""Recover short ByteTrack fragments with conservative one-to-one spatial matching."""
import time


class Tracker:
    def __init__(self, lost_timeout=30.0, recovery_timeout=3.0):
        self.lost_timeout = lost_timeout
        self.recovery_timeout = recovery_timeout
        self._seq_map = {}
        self._tracks = {}
        self._next_seq = 1

    @staticmethod
    def _score(box, previous):
        ax, ay, bx, by = box
        cx, cy, dx, dy = previous
        area = max(0, bx-ax)*max(0, by-ay)
        other = max(0, dx-cx)*max(0, dy-cy)
        overlap = max(0, min(bx, dx)-max(ax, cx))*max(0, min(by, dy)-max(ay, cy))
        if min(area, other) <= 0 or min(area, other)/max(area, other) < 0.5:
            return 0.0
        return overlap / max(1, area+other-overlap)

    def update(self, detections, now=None):
        now = time.time() if now is None else now
        stale = {tid for tid, state in self._tracks.items() if now-state[0] > self.lost_timeout}
        for tid in stale:
            del self._tracks[tid]
        self._seq_map = {raw: tid for raw, tid in self._seq_map.items() if tid not in stale}
        assigned, used = {}, set()
        for i, detection in enumerate(detections):
            tid = self._seq_map.get(detection.get('track_id', -1))
            if tid is not None and tid not in used:
                assigned[i] = tid
                used.add(tid)
        candidates = []
        for i, detection in enumerate(detections):
            if i in assigned or detection.get('track_id', -1) < 0:
                continue
            for tid, (last, box) in self._tracks.items():
                if tid in used or not 0 <= now-last <= self.recovery_timeout:
                    continue
                score = self._score(detection['xyxy'], box)
                if score >= 0.45:
                    candidates.append((score, i, tid))
        for score, i, tid in sorted(candidates, reverse=True):
            if i not in assigned and tid not in used:
                assigned[i] = tid
                used.add(tid)
        for i, detection in enumerate(detections):
            raw = detection.get('track_id', -1)
            if raw < 0:
                continue
            tid = assigned.get(i)
            if tid is None:
                tid = self._next_seq
                self._next_seq += 1
            # Remove obsolete aliases so a revived old raw ID cannot steal this track.
            self._seq_map = {key: value for key, value in self._seq_map.items() if value != tid or key == raw}
            self._seq_map[raw] = tid
            self._tracks[tid] = (now, tuple(detection['xyxy']))
            detection['track_id'] = tid
        return detections

    def reset(self):
        self._seq_map.clear()
        self._tracks.clear()
        self._next_seq = 1
