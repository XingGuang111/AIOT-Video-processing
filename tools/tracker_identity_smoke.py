"""Regression checks for short fragments and simultaneous identities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from InsightfaceModel.tracker import Tracker
from VideoTag.frame_annotations import FrameAnnotations

def person(raw, x=0):
    return {'track_id':raw, 'xyxy':(x, 0, x+100, 200)}

t = Tracker()
assert t.update([person(10)], now=0)[0]['track_id'] == 1
assert t.update([person(11, 5)], now=1)[0]['track_id'] == 1
assert [p['track_id'] for p in t.update([person(11, 10), person(12, 20)], now=2)] == [1, 2]
assert t.update([person(13, 500)], now=2.1)[0]['track_id'] == 3
assert t.update([person(14, 10)], now=6)[0]['track_id'] == 4
assert t.update([person(15, 10)], now=40)[0]['track_id'] == 5
samples = [{'index':i, 'faces':[{'face_id':99-i, 'tracking_id':7, 'person_id':1, 'xyxy':[i,0,i+10,10]}]} for i in (0,10)]
face = FrameAnnotations(samples).at(5,{})['faces'][0]
assert face['face_id'] == 0 and face['xyxy'][0] == 5
print('PASS: fragment recovery, simultaneous people, distant arrivals, expiry, frame-local face labels and interpolation')
