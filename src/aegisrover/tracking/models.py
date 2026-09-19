"""Data structures for the multiple-hypothesis tracker.

A single Kalman estimate is a *commitment*: when a detection is ambiguous or the
target disappears behind an occluder, picking one explanation immediately is what
makes the estimate jump to the wrong object. The tracker therefore keeps several
:class:`Hypothesis` branches alive per :class:`Track`, each carrying its own
posterior weight, and only lets a single interpretation dominate once the evidence
is sustained. Every branch remembers its lineage so the final decision can be
explained afterwards.
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = (
    'TrackStatus', 'Detection', 'Occluder', 'Hypothesis', 'Track',
    'HYPOTHESIS_IDS', 'TRACK_IDS', 'TAG_IDS',
)

HYPOTHESIS_IDS = itertools.count(1)
TRACK_IDS = itertools.count(1)
TAG_IDS = itertools.count(1)


class TrackStatus(str, Enum):
    """Lifecycle of a tracked object.

    TENTATIVE  births from unassociated detections; a few corroborating hits
               promote it to CONFIRMED.
    CONFIRMED  actively observed and believed to exist.
    OCCLUDED   recently confirmed, currently unobserved but its predicted position
               is explained by a known occluder — the track keeps coasting instead
               of being killed.
    LOST       unobserved beyond the miss budget without an occlusion explanation.
               The track is coasted briefly so a late reappearance can still be
               matched, then it is deleted.
    """

    TENTATIVE = 'tentative'
    CONFIRMED = 'confirmed'
    OCCLUDED = 'occluded'
    LOST = 'lost'


@dataclass(frozen=True)
class Detection:
    """A 2-D point observation. ``covariance`` is an optional 2x2 measurement R."""

    x: float
    y: float
    covariance: np.ndarray | None = None
    source: str = 'detector'

    def z(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class Occluder:
    """A convex or simple polygon that can hide a target.

    A miss is *expected* while the predicted position is covered, so the miss
    branch keeps a high posterior weight and the track enters OCCLUDED instead of
    being abandoned. ``label`` is quoted verbatim in the loss explanation.
    """

    polygon: tuple[tuple[float, float], ...]
    label: str = 'occluder'

    def covers(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def blocks(self, start, end) -> bool:
        return segment_crosses_polygon(start, end, self.polygon)


def point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    """Ray-casting test; works for any simple polygon (convex or concave)."""
    inside = False
    if len(polygon) < 3:
        return False
    ax, ay = polygon[-1]
    for bx, by in polygon:
        crosses = ((by > y) != (ay > y)) and (
            x < (ax - bx) * (y - by) / (ay - by) + bx)
        if crosses:
            inside = not inside
        ax, ay = bx, by
    return inside


def _segments_intersect(p, q, a, b) -> bool:
    def orientation(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    def on_segment(u, v, w):
        return (min(u[0], w[0]) <= v[0] <= max(u[0], w[0])
                and min(u[1], w[1]) <= v[1] <= max(u[1], w[1]))

    o1, o2 = orientation(p, q, a), orientation(p, q, b)
    o3, o4 = orientation(a, b, p), orientation(a, b, q)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    if o1 == 0 and on_segment(p, a, q):
        return True
    if o2 == 0 and on_segment(p, b, q):
        return True
    if o3 == 0 and on_segment(a, p, b):
        return True
    if o4 == 0 and on_segment(a, q, b):
        return True
    return False


def segment_crosses_polygon(start, end, polygon: tuple[tuple[float, float], ...]) -> bool:
    """True when the open segment start->end passes through the polygon interior.

    Endpoints that themselves lie on/inside the polygon are not counted as a
    crossing (the object emerging at the occluder's near edge is legal).
    """
    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))
    vertices = list(polygon)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if _segments_intersect(start, end, a, b):
            # An intersection exactly at the target's own (covered) start point
            # is expected and must not be treated as a crossing.
            if math.dist(start, a) < 1e-9 or math.dist(start, b) < 1e-9:
                continue
            return True
    return False


@dataclass
class Hypothesis:
    """One competing interpretation of a track's state.

    State ``x`` is ``[px, py, vx, vy]``. ``tag`` is inherited along a lineage and
    identifies the *interpretation* (e.g. "matched detection cluster A") even
    though a fresh hypothesis id is minted on every measurement update, which lets
    the tracker count how long the same explanation has dominated.
    """

    x: np.ndarray
    P: np.ndarray
    log_weight: float
    tag: int
    birth_frame: int
    parent_id: int | None = None
    id: int = field(default_factory=lambda: next(HYPOTHESIS_IDS))
    last_measurement_frame: int = -1
    hits: int = 0
    misses: int = 0
    last_mahalanobis: float | None = None
    origin: str = 'birth'  # birth | measurement | coast
    last_detection: tuple[float, float] | None = None
    last_detection_index: int | None = None

    def clone(self, **overrides) -> 'Hypothesis':
        fields_ = dict(x=self.x.copy(), P=self.P.copy(), log_weight=self.log_weight,
                       tag=self.tag, birth_frame=self.birth_frame, parent_id=self.id,
                       last_measurement_frame=self.last_measurement_frame, hits=self.hits,
                       misses=self.misses, last_mahalanobis=self.last_mahalanobis,
                       origin=self.origin, last_detection=self.last_detection,
                       last_detection_index=self.last_detection_index)
        fields_.update(overrides)
        return Hypothesis(**fields_)


@dataclass
class Track:
    """A tracked object together with all of its competing hypotheses."""

    created_frame: int
    hypotheses: list[Hypothesis]
    id: int = field(default_factory=lambda: next(TRACK_IDS))
    status: TrackStatus = TrackStatus.TENTATIVE
    best_tag: int | None = None
    dominance_streak: int = 0
    coast_start_frame: int | None = None
    coast_start_velocity: np.ndarray | None = None
    miss_streak: int = 0
    unexplained_misses: int = 0
    last_covered_frame: int | None = None
    hit_streak: int = 0
    committed_tag: int | None = None
    pending: dict | None = None
    best_det_index: int | None = None
    last_occluder: str | None = None
    last_loss_frame: int | None = None
    events: deque = field(default_factory=lambda: deque(maxlen=8))

    # -- derived accessors -----------------------------------------------------
    def best(self) -> Hypothesis:
        return max(self.hypotheses, key=lambda h: h.log_weight)

    def weights(self) -> np.ndarray:
        log_w = np.array([h.log_weight for h in self.hypotheses], dtype=float)
        log_w -= _logsumexp(log_w)
        return np.exp(log_w)

    def age(self, frame: int) -> int:
        return frame - self.created_frame


def _logsumexp(log_values: np.ndarray) -> float:
    peak = float(np.max(log_values))
    if not np.isfinite(peak):
        return peak
    return peak + float(np.log(np.sum(np.exp(log_values - peak))))
