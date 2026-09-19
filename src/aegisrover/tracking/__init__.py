"""Multi-hypothesis target tracking with deferred commitment.

Public entry points:

* :class:`Tracker` / :class:`TrackerConfig` -- the per-frame tracker;
* :class:`Detection`, :class:`Occluder` -- inputs;
* :class:`TrackStatus` -- tentative / confirmed / occluded / lost lifecycle;
* :class:`ConvergenceRecord`, :class:`LossRecord` -- explanations for *why* a
  particular reappearance was chosen and *why* a target was declared lost.
"""
from .explanations import (
    AssociationRecord,
    ConvergenceRecord,
    LossRecord,
    TrackFrameReport,
)
from .models import Detection, Hypothesis, Occluder, Track, TrackStatus
from .tracker import CHI2_95_2DF, Tracker, TrackerConfig

__all__ = (
    'Tracker', 'TrackerConfig', 'CHI2_95_2DF',
    'Detection', 'Occluder', 'Track', 'Hypothesis', 'TrackStatus',
    'AssociationRecord', 'ConvergenceRecord', 'LossRecord', 'TrackFrameReport',
)
