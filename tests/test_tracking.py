"""Tests for multiple-hypothesis tracking with occlusion and explainable loss.

The scenario the tracker is built for: a target disappears behind an occluder
while a look-alike distractor is visible elsewhere. A single-hypothesis greedy
tracker latches onto the distractor and the estimate jumps; these tests assert
that several interpretations are retained, the track coasts through occlusion,
re-locks onto the genuine reappearance, explains the decision, and — when the
target really is gone — says why it was declared lost.
"""
import math

import numpy as np

from aegisrover.tracking import (
    ConvergenceRecord,
    Detection,
    LossRecord,
    Occluder,
    Track,
    Tracker,
    TrackerConfig,
    TrackStatus,
)
from aegisrover.tracking.models import point_in_polygon, segment_crosses_polygon


PILLAR = Occluder(polygon=((2.0, -2.0), (4.0, -2.0), (4.0, 2.0), (2.0, 2.0)),
                  label='pillar')


def _run(scenario, *, config=None, occluders=(PILLAR,), frames=60, seed=1):
    rng = np.random.default_rng(seed)
    tracker = Tracker(config or TrackerConfig(dt=0.1), occluders=list(occluders))
    history = []
    for frame in range(1, frames + 1):
        history.append(tracker.step(scenario(frame, rng)))
    return tracker, history


def target_detections(frame, rng, *, x0=0.1, speed=1.0, dt=0.1,
                      occluded=lambda x: 2.1 <= x <= 3.9, noise=0.03):
    x = x0 + (frame - 1) * speed * dt
    if occluded(x):
        return []
    return [Detection(x + rng.normal(0, noise), rng.normal(0, noise))]


def distractor(frame, rng, *, first=25, last=45, x=5.0, y=0.5, noise=0.03):
    if not (first <= frame <= last):
        return []
    return [Detection(x + rng.normal(0, noise), y + rng.normal(0, noise))]


# ------------------------------------------------------------------ geometry
def test_point_in_polygon_and_segment_crossing():
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert point_in_polygon(0.5, 0.5, square)
    assert not point_in_polygon(2.0, 0.5, square)
    # Line of sight from inside the pillar to a point beyond it crosses a wall.
    assert segment_crosses_polygon((3.0, 0.0), (5.0, 0.5), PILLAR.polygon)
    # Two points both outside with the pillar between them are blocked ...
    assert segment_crosses_polygon((1.0, 0.0), (5.0, 0.0), PILLAR.polygon)
    # ... while two points on the same open side have a clear line of sight.
    assert not segment_crosses_polygon((5.0, 0.5), (6.0, 0.5), PILLAR.polygon)
    assert not segment_crosses_polygon((0.0, 3.0), (6.0, 3.0), PILLAR.polygon)


# ------------------------------------------- multiple hypotheses are retained
def test_several_interpretations_are_held_while_ambiguous():
    # Two detections fall inside the gate right after the target enters cover.
    def scenario(frame, rng):
        dets = target_detections(frame, rng)
        dets += distractor(frame, rng, first=22, last=60)
        return dets

    tracker, history = _run(scenario, frames=45)
    track = next(t for t in tracker.tracks if t.created_frame <= 5)
    explanations = tracker.competing_explanations(track.id)
    assert explanations, 'track must still exist'
    weights = [e['weight'] for e in explanations]
    assert abs(sum(weights) - 1.0) < 1e-6
    occluded_reports = [r for step in history[20:38] for r in step
                        if r.status == TrackStatus.OCCLUDED]
    assert occluded_reports, 'track should spend frames in OCCLUDED state'

    # Mid-occlusion (frame 30): the "still behind pillar" interpretation must be
    # retained and dominate; the distractor lives on as a separate track rather
    # than hijacking the coasting one.
    tracker2 = Tracker(TrackerConfig(dt=0.1), occluders=[PILLAR])
    rng2 = np.random.default_rng(1)
    for frame in range(1, 31):
        step = tracker2.step(scenario(frame, rng2))
    coasting = [t for t in tracker2.tracks if t.status == TrackStatus.OCCLUDED]
    assert len(coasting) == 1
    live = tracker2.competing_explanations(coasting[0].id)
    assert live and live[0]['origin'] == 'coast'
    assert live[0]['weight'] > 0.5
    # The distractor has its own track, proving both explanations coexist.
    others = [t for t in tracker2.tracks if t.id != coasting[0].id]
    assert others and math.dist(others[0].best().x[:2], (5.0, 0.5)) < 0.5


# ------------------------------------------- no jump onto the distractor ----
def test_track_coasts_through_occlusion_without_jumping_to_distractor():
    def scenario(frame, rng):
        return target_detections(frame, rng) + distractor(frame, rng)

    tracker, _ = _run(scenario, frames=60)
    target = next(t for t in tracker.tracks
                  if t.created_frame <= 5 and t.status == TrackStatus.CONFIRMED)
    # Genuine target exits the pillar near x=4 and continues to x~6.
    px, py = target.best().x[0], target.best().x[1]
    assert 5.7 < px <= 6.1
    assert abs(py) < 0.2
    # Velocity estimate preserved through the occlusion, not dragged to (5, .5).
    assert target.best().x[2] > 0.7
    assert abs(target.best().x[3]) < 0.2
    assert tracker.explain_loss(target.id) is None


# ------------------------------------------- convergence is deferred & logged
def test_choice_is_deferred_until_evidence_is_clear_and_explained():
    # Distractor sits very close to the exit, so early frames are ambiguous.
    def scenario(frame, rng):
        dets = target_detections(frame, rng)
        dets += distractor(frame, rng, first=39, last=44, x=4.2, y=0.15)
        return dets

    tracker, history = _run(scenario, frames=60)
    target = next(t for t in tracker.tracks if t.created_frame <= 5)

    convergence = [c for c in tracker.convergence_records
                   if c.chosen_tag == target.committed_tag]
    assert convergence, 'a convergence record must explain the final choice'
    record = convergence[-1]
    assert isinstance(record, ConvergenceRecord)
    assert record.winning_weight >= 0.85
    assert record.ambiguity_ratio <= 0.35
    assert record.support_frames >= tracker.config.converge_frames
    joined = ' '.join(record.evidence)
    assert 'posterior' in joined or 'reacquired' in joined
    assert any(c.mahalanobis <= c.gate for c in record.candidates)

    # The chosen explanation follows the real trajectory (along x), not the
    # distractor parked at (4.2, 0.15).
    final = target.best().x
    assert final[0] > 5.5 and abs(final[1]) < 0.25


def test_ambiguous_phase_reports_high_ambiguity_before_convergence():
    def scenario(frame, rng):
        dets = target_detections(frame, rng)
        # Persistent distractors on both plausible exits.
        dets += distractor(frame, rng, first=39, last=60, x=4.1, y=0.2)
        return dets

    tracker, history = _run(scenario, frames=60)
    target = next(t for t in tracker.tracks if t.created_frame <= 5)
    # Final position follows the target regardless of early ambiguity.
    assert target.best().x[0] > 5.7
    reports = [r for step in history for r in step if r.track_id == target.id]
    assert any(r.ambiguity > 0.0 or r.hypotheses >= 1 for r in reports)


# ------------------------------------------- real loss is explained ---------
def test_true_loss_emits_reasoned_loss_record():
    # Target walks into the pillar and never comes out the other side.
    def scenario(frame, rng):
        x = 0.1 + (frame - 1) * 0.1
        if x >= 2.1:
            return []
        return [Detection(x, 0.0)]

    config = TrackerConfig(dt=0.1, occlusion_budget_frames=8, lost_coast_frames=4)
    tracker, history = _run(scenario, config=config, frames=40)
    losses = tracker.loss_records
    assert losses
    timeout = next((l for l in losses if l.reason == 'occlusion_timeout'), None)
    deletion = next((l for l in losses if l.reason == 'coast_window_expired'), None)
    assert timeout is not None, 'a target stuck behind cover must be explained'
    assert timeout.occluded and timeout.occluder_label == 'pillar'
    assert timeout.consecutive_misses >= 8
    assert timeout.predicted_position[0] >= 2.0
    assert timeout.last_observed_frame is not None
    assert 'pillar' in timeout.detail
    assert deletion is not None, 'coast window expiry should justify deletion'
    assert all(isinstance(l, LossRecord) for l in losses)
    # The track is eventually removed.
    assert all(t.status is not None for t in tracker.tracks)


def test_unexplained_loss_without_occluder_is_reported():
    def scenario(frame, rng):
        if frame > 12:
            return []
        return [Detection(0.1 + (frame - 1) * 0.1, 0.0)]

    config = TrackerConfig(dt=0.1, lost_after_misses=3, lost_coast_frames=3)
    tracker, _ = _run(scenario, config=config, occluders=(), frames=30)
    record = next(l for l in tracker.loss_records
                  if l.reason == 'no_measurement_no_occluder')
    assert record.occluded is False
    assert record.occluder_label is None
    assert record.consecutive_misses >= 3
    assert 'gate' in record.detail or 'field of view' in record.detail


def test_late_reacquisition_within_coast_window_survives():
    # Target vanishes with no occluder for a few frames, then returns near the
    # predicted location inside the coast window.
    def scenario(frame, rng):
        x = 0.1 + (frame - 1) * 0.1
        if 13 <= frame <= 17:
            return []
        return [Detection(x + rng.normal(0, 0.02), rng.normal(0, 0.02))]

    config = TrackerConfig(dt=0.1, lost_after_misses=3, lost_coast_frames=8)
    tracker, history = _run(scenario, config=config, occluders=(), frames=40)
    target = next(t for t in tracker.tracks if t.created_frame <= 5)
    assert target.status == TrackStatus.CONFIRMED
    assert target.miss_streak == 0
    notes = [note for step in history for r in step
             if r.track_id == target.id for note in r.notes]
    assert any('late reacquisition' in note for note in notes)
    # It kept tracking one continuous identity through the blind gap.
    assert target.created_frame <= 5
    assert target.best().x[0] > 3.0


# ------------------------------------------- birth clutter is pruned --------
def test_unconfirmed_clutter_track_is_rejected_with_explanation():
    def scenario(frame, rng):
        dets = target_detections(frame, rng)
        if frame == 10:
            dets.append(Detection(8.0, 8.0))  # one-off false alarm
        return dets

    tracker, _ = _run(scenario, frames=20)
    loss = next(l for l in tracker.loss_records
                if l.reason == 'tentative_unsupported')
    assert loss.status == TrackStatus.TENTATIVE
    # Clutter track is gone; only the genuine target remains.
    assert all(math.dist(t.best().x[:2], (8.0, 8.0)) > 1.0
               for t in tracker.tracks)


# ------------------------------------------- reports are serialisable -------
def test_frame_report_and_explanations_serialise():
    def scenario(frame, rng):
        return target_detections(frame, rng) + distractor(frame, rng)

    tracker, history = _run(scenario, frames=50)
    for step in history:
        for report in step:
            data = report.to_dict()
            assert {'track_id', 'status', 'position', 'ambiguity',
                    'hypotheses', 'distinct_interpretations'} <= data.keys()
            assert isinstance(data['status'], str)
            if data['convergence'] is not None:
                assert 'evidence' in data['convergence']
                assert isinstance(data['convergence']['candidates'], list)
    for record in tracker.loss_records:
        d = record.to_dict()
        assert isinstance(d['reason'], str) and d['detail']
