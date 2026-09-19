# AegisRover

Autonomous mobile robot simulation, navigation, estimation, control, power, mission, protocol and runtime platform.

## Multi-hypothesis target tracking

`aegisrover.tracking` keeps several competing interpretations of a target alive
instead of committing to one greedy association per frame. When a target is
occluded, the track coasts (with an occluder-aware miss model and line-of-sight
gating) while reappearances and distractors compete as weighted hypotheses; the
tracker only converges after the evidence is sustained and records why. Real
losses produce a structured `LossRecord` explaining the reason (no occluder,
occlusion timeout, coast window expired).

```python
from aegisrover.tracking import Tracker, TrackerConfig, Detection, Occluder

tracker = Tracker(TrackerConfig(dt=0.1), occluders=[Occluder(polygon, 'pillar')])
for detections in frames:
    reports = tracker.step(detections)
tracker.competing_explanations(track_id)  # weighted live alternatives
tracker.explain_loss(track_id)            # LossRecord | None
```
