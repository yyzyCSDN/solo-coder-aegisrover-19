"""Structured explanations for tracker decisions.

The tracker must answer two questions after the fact:

* why did it converge on *this* reappearance rather than the nearby distractor;
* when a target is declared lost, what evidence (or missing evidence) justifies it.

These dataclasses are the machine-readable answers; ``to_dict`` mirrors the style
used elsewhere in the codebase (rounded floats, string reasons) so they can be
serialised straight into a run record or operator UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import TrackStatus

__all__ = ('AssociationRecord', 'ConvergenceRecord', 'LossRecord', 'TrackFrameReport')


@dataclass(frozen=True)
class AssociationRecord:
    """One candidate that was considered when the track reappeared."""

    detection_index: int
    mahalanobis: float
    gate: float
    inside_gate: bool
    log_likelihood: float
    weight_after: float

    def to_dict(self) -> dict:
        return {'detection_index': self.detection_index,
                'mahalanobis': round(self.mahalanobis, 4),
                'gate': round(self.gate, 4),
                'inside_gate': self.inside_gate,
                'log_likelihood': round(self.log_likelihood, 4),
                'weight_after': round(self.weight_after, 6)}


@dataclass(frozen=True)
class ConvergenceRecord:
    """Why the surviving interpretation was chosen over its competitors."""

    frame: int
    chosen_tag: int
    winning_weight: float
    support_frames: int
    margin: float
    ambiguity_ratio: float
    evidence: tuple[str, ...]
    candidates: tuple[AssociationRecord, ...]

    def to_dict(self) -> dict:
        return {'frame': self.frame, 'chosen_tag': self.chosen_tag,
                'winning_weight': round(self.winning_weight, 6),
                'support_frames': self.support_frames,
                'margin': round(self.margin, 6),
                'ambiguity_ratio': round(self.ambiguity_ratio, 4),
                'evidence': list(self.evidence),
                'candidates': [c.to_dict() for c in self.candidates]}


@dataclass(frozen=True)
class LossRecord:
    """Why a track was declared lost (or why it is merely occluded)."""

    track_id: int
    frame: int
    status: TrackStatus
    reason: str
    consecutive_misses: int
    miss_budget: int
    occluded: bool
    occluder_label: str | None
    predicted_position: tuple[float, float]
    last_observed_position: tuple[float, float] | None
    last_observed_frame: int | None
    nearest_detection_mahalanobis: float | None
    detail: str

    def to_dict(self) -> dict:
        return {'track_id': self.track_id, 'frame': self.frame,
                'status': self.status.value, 'reason': self.reason,
                'consecutive_misses': self.consecutive_misses,
                'miss_budget': self.miss_budget,
                'occluded': self.occluded,
                'occluder_label': self.occluder_label,
                'predicted_position': tuple(round(v, 4) for v in self.predicted_position),
                'last_observed_position': (None if self.last_observed_position is None
                                           else tuple(round(v, 4) for v in self.last_observed_position)),
                'last_observed_frame': self.last_observed_frame,
                'nearest_detection_mahalanobis': (None
                    if self.nearest_detection_mahalanobis is None
                    else round(self.nearest_detection_mahalanobis, 4)),
                'detail': self.detail}


@dataclass
class TrackFrameReport:
    """Everything an operator should see about one track on one frame."""

    track_id: int
    status: TrackStatus
    position: tuple[float, float]
    velocity: tuple[float, float]
    best_weight: float
    hypotheses: int
    distinct_interpretations: int
    consecutive_misses: int
    ambiguity: float
    convergence: ConvergenceRecord | None = None
    loss: LossRecord | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'track_id': self.track_id, 'status': self.status.value,
                'position': tuple(round(v, 4) for v in self.position),
                'velocity': tuple(round(v, 4) for v in self.velocity),
                'best_weight': round(self.best_weight, 6),
                'hypotheses': self.hypotheses,
                'distinct_interpretations': self.distinct_interpretations,
                'consecutive_misses': self.consecutive_misses,
                'ambiguity': round(self.ambiguity, 4),
                'convergence': None if self.convergence is None else self.convergence.to_dict(),
                'loss': None if self.loss is None else self.loss.to_dict(),
                'notes': list(self.notes)}
