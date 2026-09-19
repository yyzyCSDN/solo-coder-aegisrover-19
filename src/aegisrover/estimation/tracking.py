"""Multi-hypothesis tracking through occlusion and re-acquisition.

A single-estimate tracker must decide *immediately* which detection is the
target. When the target reappears from behind an occlusion next to clutter,
that decision is a coin flip — and a Kalman filter cannot take it back, so
the track snaps to the wrong position and the true reappearance is gated out
for good. This tracker instead keeps several competing interpretations alive
at once: each hypothesis is a Kalman filter with its own association history
and probability, and the tracker only commits when the evidence clearly
favours one of them. Commitments — and declarations that the track is lost —
come with the evidence behind them, so an operator can always ask *why*.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .filters import KalmanFilter, mahalanobis_squared

__all__ = (
    'TRACKING', 'AMBIGUOUS', 'COASTING', 'LOST',
    'ConstantVelocityModel', 'AssociationEvent', 'TrackEstimate',
    'ChoiceExplanation', 'LostExplanation', 'ScanReport', 'HypothesisTracker',
)

TRACKING = 'tracking'
AMBIGUOUS = 'ambiguous'
COASTING = 'coasting'
LOST = 'lost'


class ConstantVelocityModel:
    """2-D constant-velocity model: state ``[x, y, vx, vy]``, position detections.

    ``process_noise`` is the spectral density of the white acceleration noise —
    how far the velocity can wander per second — and ``measurement_noise`` the
    standard deviation of a position detection, both in metres.
    """

    def __init__(self, process_noise: float = 0.5, measurement_noise: float = 0.25):
        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError('noise levels must be positive')
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.diag([self.measurement_noise ** 2] * 2)

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Discretised transition and process noise for a step of ``dt`` seconds."""
        if dt <= 0:
            raise ValueError('dt must be positive')
        dt2 = dt * dt
        q = self.process_noise ** 2
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        Q = q * np.array([[dt2 * dt / 3.0, 0.0, dt2 / 2.0, 0.0],
                          [0.0, dt2 * dt / 3.0, 0.0, dt2 / 2.0],
                          [dt2 / 2.0, 0.0, dt, 0.0],
                          [0.0, dt2 / 2.0, 0.0, dt]])
        return F, Q


@dataclass(frozen=True)
class AssociationEvent:
    """One entry in a hypothesis's association history.

    ``kind`` is ``'spawned'`` (the hypothesis was created from an otherwise
    unexplained detection), ``'associated'`` (a detection fell inside the gate
    and was folded in) or ``'missed'`` (no detection matched this scan, e.g.
    because the target was occluded). ``distance`` is the squared Mahalanobis
    distance of the association and is only set for ``'associated'`` events.
    """
    time: float
    kind: str
    distance: float | None = None

    def to_dict(self) -> dict:
        return {'time': round(self.time, 6), 'kind': self.kind,
                'distance': None if self.distance is None else round(self.distance, 6)}


@dataclass(frozen=True)
class TrackEstimate:
    """The tracker's best current answer, tagged with how much to trust it."""
    hypothesis_id: int
    time: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    position_covariance: tuple[tuple[float, float], tuple[float, float]]
    probability: float
    consecutive_hits: int
    scans_since_update: int

    def to_dict(self) -> dict:
        return {'hypothesis_id': self.hypothesis_id, 'time': round(self.time, 6),
                'position': [round(v, 6) for v in self.position],
                'velocity': [round(v, 6) for v in self.velocity],
                'position_covariance': [[round(v, 6) for v in row] for row in self.position_covariance],
                'probability': round(self.probability, 6),
                'consecutive_hits': self.consecutive_hits,
                'scans_since_update': self.scans_since_update}


@dataclass(frozen=True)
class ChoiceExplanation:
    """Why the tracker committed to this hypothesis and not the alternatives."""
    hypothesis_id: int
    probability: float
    runner_up_id: int | None
    runner_up_probability: float | None
    margin: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {'hypothesis_id': self.hypothesis_id, 'probability': round(self.probability, 6),
                'runner_up_id': self.runner_up_id,
                'runner_up_probability': (None if self.runner_up_probability is None
                                          else round(self.runner_up_probability, 6)),
                'margin': None if self.margin is None else round(self.margin, 6),
                'reasons': list(self.reasons)}


@dataclass(frozen=True)
class LostExplanation:
    """Why the tracker declared the track lost, and what it knew at the time."""
    reason: str
    detail: str
    scans_without_update: int
    last_association: AssociationEvent | None
    last_position: tuple[float, float] | None

    def to_dict(self) -> dict:
        return {'reason': self.reason, 'detail': self.detail,
                'scans_without_update': self.scans_without_update,
                'last_association': (None if self.last_association is None
                                     else self.last_association.to_dict()),
                'last_position': (None if self.last_position is None
                                  else [round(v, 6) for v in self.last_position])}


@dataclass(frozen=True)
class ScanReport:
    """Outcome of one scan: status, best estimate and the explanations."""
    time: float
    status: str
    detections: int
    hypotheses: int
    best: TrackEstimate | None
    choice: ChoiceExplanation | None
    lost: LostExplanation | None
    probabilities: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict:
        return {'time': round(self.time, 6), 'status': self.status,
                'detections': self.detections, 'hypotheses': self.hypotheses,
                'best': None if self.best is None else self.best.to_dict(),
                'choice': None if self.choice is None else self.choice.to_dict(),
                'lost': None if self.lost is None else self.lost.to_dict(),
                'probabilities': [{'id': ident, 'probability': round(p, 6)}
                                  for ident, p in self.probabilities]}


class _Hypothesis:
    """One interpretation of the detection history: a filter plus its evidence."""

    __slots__ = ('id', 'kf', 'log_score', 'hits', 'consecutive_hits',
                 'scans_since_update', 'born', 'last_update_time', 'events')

    def __init__(self, ident: int, kf: KalmanFilter, log_score: float, born: float, history: int):
        self.id = ident
        self.kf = kf
        self.log_score = log_score
        self.hits = 0
        self.consecutive_hits = 0
        self.scans_since_update = 0
        self.born = born
        self.last_update_time = born
        self.events: deque[AssociationEvent] = deque(maxlen=history)

    def clone(self, ident: int) -> '_Hypothesis':
        other = _Hypothesis(ident, KalmanFilter(self.kf.x.copy(), self.kf.P.copy()),
                            self.log_score, self.born, self.events.maxlen)
        other.hits = self.hits
        other.consecutive_hits = self.consecutive_hits
        other.scans_since_update = self.scans_since_update
        other.last_update_time = self.last_update_time
        other.events = deque(self.events, maxlen=self.events.maxlen)
        return other


class HypothesisTracker:
    """Tracks one target by keeping several data-association interpretations alive.

    Every scan, each hypothesis branches: it considers every detection inside
    its gate *and* the possibility that the target was not detected at all
    (occlusion). Detections no hypothesis can explain spawn fresh
    re-acquisition candidates. Branches are scored by likelihood, normalised
    to probabilities and pruned, so competing explanations survive until the
    evidence separates them. The tracker reports:

    - ``TRACKING`` — one hypothesis is both probable and repeatedly confirmed;
      ``explain_choice()`` says why it won.
    - ``AMBIGUOUS`` — interpretations still compete; no commitment is made.
    - ``COASTING`` — no detection matched recently; extrapolating, not yet lost.
    - ``LOST`` — every hypothesis went too long without a detection;
      ``explain_lost()`` says why.
    """

    def __init__(self, model: ConstantVelocityModel | None = None, *, gate: float = 9.21,
                 detection_probability: float = 0.9, clutter_density: float = 1e-3,
                 spawn_weight: float = 0.05, confirm_probability: float = 0.6,
                 min_probability: float = 0.01, max_hypotheses: int = 8,
                 max_coast_scans: int = 5, min_hits: int = 2, history: int = 12,
                 spawn_velocity_sigma: float = 2.0, merge_tolerance: float = 0.3):
        self.model = model or ConstantVelocityModel()
        if gate <= 0:
            raise ValueError('gate must be positive')
        for name, value in (('detection_probability', detection_probability),
                            ('spawn_weight', spawn_weight),
                            ('confirm_probability', confirm_probability),
                            ('min_probability', min_probability)):
            if not 0.0 < value < 1.0:
                raise ValueError(f'{name} must be in (0, 1)')
        if clutter_density <= 0:
            raise ValueError('clutter_density must be positive')
        if max_hypotheses < 1:
            raise ValueError('max_hypotheses must be at least 1')
        if max_coast_scans < 1:
            raise ValueError('max_coast_scans must be at least 1')
        if min_hits < 1:
            raise ValueError('min_hits must be at least 1')
        if history < 1:
            raise ValueError('history must be at least 1')
        if spawn_velocity_sigma <= 0:
            raise ValueError('spawn_velocity_sigma must be positive')
        if merge_tolerance <= 0:
            raise ValueError('merge_tolerance must be positive')
        self.gate = float(gate)
        self.detection_probability = float(detection_probability)
        self.clutter_density = float(clutter_density)
        self.spawn_weight = float(spawn_weight)
        self.confirm_probability = float(confirm_probability)
        self.min_probability = float(min_probability)
        self.max_hypotheses = int(max_hypotheses)
        self.max_coast_scans = int(max_coast_scans)
        self.min_hits = int(min_hits)
        self.history = int(history)
        self.spawn_velocity_sigma = float(spawn_velocity_sigma)
        self.merge_tolerance = float(merge_tolerance)
        self._hypotheses: list[_Hypothesis] = []
        self._next_id = 1
        self._time: float | None = None
        self._start_time: float | None = None
        self._scans_without_update = 0
        self._last_association: AssociationEvent | None = None
        self._last_position: tuple[float, float] | None = None
        self._status = LOST
        self._choice: ChoiceExplanation | None = None
        self._lost: LostExplanation | None = None

    # -- public interface ------------------------------------------------------
    @property
    def status(self) -> str:
        return self._status

    @property
    def hypothesis_count(self) -> int:
        return len(self._hypotheses)

    def start(self, position: Sequence[float], time: float,
              velocity: Sequence[float] = (0.0, 0.0)) -> None:
        """Seed a single certain hypothesis at ``position``; replaces any existing ones."""
        x = [float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1])]
        P = np.diag([self.model.R[0, 0], self.model.R[1, 1],
                     self.spawn_velocity_sigma ** 2, self.spawn_velocity_sigma ** 2])
        hypothesis = _Hypothesis(self._fresh_id(), KalmanFilter(x, P), 0.0, float(time), self.history)
        hypothesis.events.append(AssociationEvent(float(time), 'spawned'))
        self._hypotheses = [hypothesis]
        self._time = float(time)
        self._start_time = float(time)
        self._scans_without_update = 0
        self._last_association = None
        self._last_position = None
        self._status = AMBIGUOUS
        self._choice = None
        self._lost = None

    def scan(self, detections: Sequence[Sequence[float]], time: float) -> ScanReport:
        """Fold one scan of position detections into the hypotheses.

        ``detections`` is a sequence of ``(x, y)`` positions, possibly empty
        (the target was occluded or left the field of view). ``time`` must be
        strictly increasing between calls.
        """
        time = float(time)
        detections = [self._validate_detection(d) for d in detections]
        if self._time is not None:
            if time <= self._time:
                raise ValueError('scan times must be strictly increasing')
            F, Q = self.model.matrices(time - self._time)
            for hypothesis in self._hypotheses:
                hypothesis.kf.predict(F, Q)
        if self._start_time is None:
            self._start_time = time
        self._time = time

        branches, explained, updated = self._branch(detections, time)
        spawned = self._spawn(detections, explained, branches, time)
        branches.extend(spawned)
        updated = updated or bool(spawned)
        self._normalise(branches)
        self._hypotheses = self._prune(self._merge(branches))
        self._scans_without_update = 0 if updated else self._scans_without_update + 1
        self._decide_status()
        return ScanReport(time=time, status=self._status, detections=len(detections),
                          hypotheses=len(self._hypotheses), best=self.best_estimate(),
                          choice=self._choice if self._status == TRACKING else None,
                          lost=self._lost if self._status == LOST else None,
                          probabilities=self._probabilities())

    def best_estimate(self) -> TrackEstimate | None:
        """The most probable hypothesis, or ``None`` when the track is lost."""
        if not self._hypotheses:
            return None
        return self._estimate(self._best())

    def explain_choice(self) -> ChoiceExplanation | None:
        """Why the current winning hypothesis was chosen (``TRACKING`` only)."""
        return self._choice if self._status == TRACKING else None

    def explain_lost(self) -> LostExplanation | None:
        """Why the track was declared lost (``LOST`` only)."""
        return self._lost if self._status == LOST else None

    # -- branching -------------------------------------------------------------
    def _branch(self, detections, time):
        """Split every hypothesis into miss and association branches."""
        branches = []
        explained = [False] * len(detections)
        updated = False
        miss_log = math.log(1.0 - self.detection_probability)
        for hypothesis in self._hypotheses:
            parent_score = hypothesis.log_score
            parent_consecutive = hypothesis.consecutive_hits
            predicted = self.model.H @ hypothesis.kf.x
            S = self.model.H @ hypothesis.kf.P @ self.model.H.T + self.model.R
            for index, z in enumerate(detections):
                distance = mahalanobis_squared(np.asarray(z) - predicted, S)
                if distance > self.gate:
                    continue
                explained[index] = True
                updated = True
                child = hypothesis.clone(self._fresh_id())
                child.kf.update(z, self.model.H, self.model.R)
                child.log_score = parent_score + self._association_log_weight(distance, S)
                child.hits += 1
                child.consecutive_hits = parent_consecutive + 1
                child.scans_since_update = 0
                child.last_update_time = time
                event = AssociationEvent(time, 'associated', distance)
                child.events.append(event)
                self._last_association = event
                self._last_position = (float(child.kf.x[0]), float(child.kf.x[1]))
                branches.append(child)
            hypothesis.log_score = parent_score + miss_log
            hypothesis.consecutive_hits = 0
            hypothesis.scans_since_update += 1
            hypothesis.events.append(AssociationEvent(time, 'missed'))
            branches.append(hypothesis)
        return branches, explained, updated

    def _spawn(self, detections, explained, branches, time):
        """Create re-acquisition candidates from detections nobody could explain."""
        if not any(not e for e in explained):
            return []
        anchor = max((b.log_score for b in branches), default=0.0)
        spawn_log = math.log(self.spawn_weight)
        spawned = []
        for index, z in enumerate(detections):
            if explained[index]:
                continue
            kf = KalmanFilter([z[0], z[1], 0.0, 0.0],
                              np.diag([self.model.R[0, 0], self.model.R[1, 1],
                                       self.spawn_velocity_sigma ** 2, self.spawn_velocity_sigma ** 2]))
            child = _Hypothesis(self._fresh_id(), kf,
                                (anchor + spawn_log) if branches else 0.0, time, self.history)
            child.hits = 1
            child.consecutive_hits = 1
            child.last_update_time = time
            event = AssociationEvent(time, 'spawned')
            child.events.append(event)
            self._last_association = event
            self._last_position = (z[0], z[1])
            spawned.append(child)
        return spawned

    def _association_log_weight(self, distance: float, S: np.ndarray) -> float:
        """Log likelihood ratio of 'detection is the target' versus 'detection is clutter'."""
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            logdet = math.log(max(float(np.linalg.det(S)), 1e-12))
        log_likelihood = -0.5 * (distance + S.shape[0] * math.log(2.0 * math.pi) + logdet)
        return math.log(self.detection_probability) + log_likelihood - math.log(self.clutter_density)

    # -- population maintenance --------------------------------------------------
    def _normalise(self, branches) -> None:
        if not branches:
            return
        top = max(b.log_score for b in branches)
        norm = top + math.log(sum(math.exp(b.log_score - top) for b in branches))
        for branch in branches:
            branch.log_score -= norm

    def _merge(self, branches):
        """Fold branches that converged to the same state back into one."""
        buckets: dict[tuple[int, ...], list[_Hypothesis]] = {}
        for branch in branches:
            key = tuple(int(round(v / self.merge_tolerance)) for v in branch.kf.x)
            buckets.setdefault(key, []).append(branch)
        merged = []
        for group in buckets.values():
            group.sort(key=lambda b: b.log_score, reverse=True)
            keeper = group[0]
            for other in group[1:]:
                keeper.log_score = float(np.logaddexp(keeper.log_score, other.log_score))
            merged.append(keeper)
        return merged

    def _prune(self, branches):
        live = [b for b in branches if b.scans_since_update <= self.max_coast_scans]
        if not live:
            return []
        floored = [b for b in live if math.exp(b.log_score) >= self.min_probability]
        live = floored or [max(live, key=lambda b: b.log_score)]
        live.sort(key=lambda b: b.log_score, reverse=True)
        survivors = live[:self.max_hypotheses]
        top = max(b.log_score for b in survivors)
        norm = top + math.log(sum(math.exp(b.log_score - top) for b in survivors))
        for survivor in survivors:
            survivor.log_score -= norm
        return survivors

    # -- status and explanations -------------------------------------------------
    def _decide_status(self) -> None:
        if not self._hypotheses:
            self._status = LOST
            self._choice = None
            self._lost = self._build_lost_explanation()
            return
        best = self._best()
        if (best.scans_since_update == 0 and best.consecutive_hits >= self.min_hits
                and math.exp(best.log_score) >= self.confirm_probability):
            self._status = TRACKING
            self._choice = self._build_choice_explanation(best)
        elif best.scans_since_update == 0 or any(h.scans_since_update == 0 for h in self._hypotheses):
            self._status = AMBIGUOUS
            self._choice = None
        else:
            self._status = COASTING
            self._choice = None

    def _build_choice_explanation(self, best: _Hypothesis) -> ChoiceExplanation:
        ordered = sorted(self._hypotheses, key=lambda h: h.log_score, reverse=True)
        runner = ordered[1] if len(ordered) > 1 else None
        probability = math.exp(best.log_score)
        runner_probability = math.exp(runner.log_score) if runner is not None else None
        margin = None if runner_probability is None else probability - runner_probability
        reasons = []
        associated = [e for e in best.events if e.kind == 'associated']
        if best.events:
            reasons.append(f'associated {len(associated)} of the last {len(best.events)} scans')
        if associated:
            reasons.append(f'latest association distance {associated[-1].distance:.2f} '
                           f'within gate {self.gate:.2f}')
        if self._start_time is not None and best.born > self._start_time:
            reasons.append(f'hypothesis first appeared at t={best.born:.3f} as a '
                           f're-acquisition candidate')
        if runner is not None:
            trailing_misses = 0
            for event in reversed(runner.events):
                if event.kind != 'missed':
                    break
                trailing_misses += 1
            if trailing_misses:
                reasons.append(f'runner-up #{runner.id} (p={runner_probability:.2f}) missed '
                               f'the last {trailing_misses} scan(s)')
            else:
                reasons.append(f'runner-up #{runner.id} (p={runner_probability:.2f}) explains '
                               f'the recent detections less well')
        if margin is not None and margin < 0.2:
            reasons.append('margin over runner-up is small; the choice may still change')
        return ChoiceExplanation(best.id, probability,
                                 None if runner is None else runner.id,
                                 runner_probability, margin, tuple(reasons))

    def _build_lost_explanation(self) -> LostExplanation:
        if self._last_association is None:
            reason = 'no_observations'
            detail = 'no detection has ever produced a hypothesis; there is nothing to track'
        else:
            reason = 'coast_exceeded'
            detail = (f'no detection within the gate of any hypothesis for '
                      f'{self._scans_without_update} consecutive scans (limit '
                      f'{self.max_coast_scans}); last association at '
                      f't={self._last_association.time:.3f}')
        return LostExplanation(reason, detail, self._scans_without_update,
                               self._last_association, self._last_position)

    # -- helpers -----------------------------------------------------------------
    def _estimate(self, hypothesis: _Hypothesis) -> TrackEstimate:
        x = hypothesis.kf.x
        P = hypothesis.kf.P
        return TrackEstimate(hypothesis_id=hypothesis.id, time=self._time,
                             position=(float(x[0]), float(x[1])),
                             velocity=(float(x[2]), float(x[3])),
                             position_covariance=((float(P[0, 0]), float(P[0, 1])),
                                                  (float(P[1, 0]), float(P[1, 1]))),
                             probability=math.exp(hypothesis.log_score),
                             consecutive_hits=hypothesis.consecutive_hits,
                             scans_since_update=hypothesis.scans_since_update)

    def _best(self) -> _Hypothesis | None:
        return max(self._hypotheses, key=lambda h: h.log_score, default=None)

    def _probabilities(self) -> tuple[tuple[int, float], ...]:
        ordered = sorted(self._hypotheses, key=lambda h: h.log_score, reverse=True)
        return tuple((h.id, math.exp(h.log_score)) for h in ordered)

    def _fresh_id(self) -> int:
        ident = self._next_id
        self._next_id += 1
        return ident

    @staticmethod
    def _validate_detection(detection) -> tuple[float, float]:
        values = tuple(map(float, detection))
        if len(values) != 2:
            raise ValueError('detections must be (x, y) pairs')
        return values
