"""Smooth sample boxes onto every output frame without mixing track identities."""
from bisect import bisect_right


class FrameAnnotations:
    def __init__(self, samples):
        self.samples = sorted(samples, key=lambda sample: sample['index'])
        self.indices = [sample['index'] for sample in self.samples]

    def at(self, index, fallback):
        if not self.samples:
            return fallback
        left_index = max(0, bisect_right(self.indices, index) - 1)
        right_index = min(left_index + 1, len(self.samples) - 1)
        left, right = self.samples[left_index], self.samples[right_index]
        span = right['index'] - left['index']
        fraction = min(1, max(0, (index - left['index']) / span)) if span else 0
        result = {}
        for kind, identity in (('persons', 'track_id'), ('faces', 'face_id')):
            def key_of(item):
                if kind == 'faces':
                    pid = item.get('person_id', 4294967295)
                    return ('person', pid) if pid != 4294967295 else ('face', item.get('tracking_id', item['face_id']))
                return item[identity]
            following = {key_of(item): item for item in right.get(kind, [])}
            records = []
            seen = set()
            for item in left.get(kind, []):
                key = key_of(item)
                if key in seen or (kind == 'persons' and key < 0):
                    continue
                seen.add(key)
                record = dict(item)
                next_item = following.get(key)
                if next_item is not None and (kind != 'faces' or item['person_id'] == next_item['person_id']):
                    record['xyxy'] = [round(a + (b - a) * fraction)
                                      for a, b in zip(item['xyxy'], next_item['xyxy'])]
                records.append(record)
            if kind == 'faces':
                records.sort(key=lambda item: item['xyxy'][0])
                for face_index, record in enumerate(records):
                    record['face_id'] = face_index
            result[kind] = records
        return result
