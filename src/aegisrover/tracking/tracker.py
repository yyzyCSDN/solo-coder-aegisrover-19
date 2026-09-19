"""Multiple-hypothesis tracker with deferred commitment and explainable losses.

The update loop deliberately keeps several explanations alive instead of forcing a
single association per frame:

1. every surviving hypothesis **predicts** its state forward;
2. each gating detection spawns a *measurement* child (hit / match);
3. a *miss* child always survives as well — with a much smaller probability penalty
   when an occluder covers the predicted position, so occlusion is treated as
   expected missing evidence rather than contradictory evidence;
4. weights are the children's Bayesian posterior probabilities; weak branches are
   pruned and branches that say the same thing (same tag and nearby state) are
   merged;
5. promotion, occlusion, loss and cross-track de-duplication only happen *after*
   the evidence has been allowed to accumulate (``confirm_hits``,
   ``converge_frames``), and every decision emits a structured record.

This prevents the classic failure: target hides, a distractor appears, a greedy
nearest-neighbour filter latches onto the distractor and the state jumps. Here
"reconnect here" and "still hidden / the target is the later reappearance" compete
on likelihood for several frames before either is committed.

Typical use::

    tracker = Tracker(TrackerConfig(), occluders=[Occluder(polygon, 'pillar')])
    for frame_detections in frames:
        reports = tracker.step(frame_detections)
        for report in reports:
            ...  # status, position, ambiguity, convergence/loss explanations
    tracker.competing_explanations(track_id)  # live alternatives
    tracker.explain_loss(track_id)            # structured LossRecord
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .explanations import (
    AssociationRecord,
    ConvergenceRecord,
    LossRecord,
    TrackFrameReport,
)
from .models import (
    TAG_IDS,
    Detection,
    Hypothesis,
    Occluder,
    Track,
    TrackStatus,
)
from .motion import measurement_covariance, predict_hypothesis, update_hypothesis

__all__ = ('TrackerConfig', 'Tracker', 'CHI2_95_2DF')

# Chi-square 95% gate for a 2-D position innovation: P(chi2_2 <= 5.991) ~= 0.95.
CHI2_95_2DF = 5.991
_OBSERVATION = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]], dtype=float)


def _point_segment_distance(point, a, b) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


@dataclass(frozen=True)
class TrackerConfig:
    """Tunables. Defaults describe a slow indoor target seen by a 2-D detector."""

    gate: float = CHI2_95_2DF
    q_sigma: float = 1.0                 # process-noise acceleration std [m/s^2]
    measurement_variance: float = 0.05   # default R when a detection gives none
    p_detect: float = 0.9                # P(seeing a visible target)
    p_detect_occluded: float = 0.15      # residual visibility of a covered target
    clutter_density: float = 0.02        # uniform false-alarm density lambda per m^2;
                                         # a true association weighs as
                                         # p_detect * (1/lambda) * exp(-d2/2)
    prune_weight: float = 1e-3           # drop hypotheses below this posterior
    max_hypotheses: int = 12
    confirm_hits: int = 3                # corroborating hits to confirm a birth
    tentative_max_misses: int = 2
    lost_after_misses: int = 3           # unexplained misses before a confirmed
                                         # target is declared LOST (it still coasts)
    reacquire_grace: int = 3             # frames of tolerance right after the
                                         # prediction leaves an occluder
    occlusion_budget_frames: int = 30    # coast this long while covered
    lost_coast_frames: int = 8           # keep matching briefly after a true loss
    converge_weight: float = 0.85
    converge_frames: int = 3
    ambiguity_ratio: float = 0.35        # second-best/best above this => ambiguous
    merge_distance: float = 0.5          # duplicate-track separation [m]
    tag_radius: float = 0.75             # branches within this share an interpretation
    emerge_distance: float = 1.0         # a sighting this close to a covered
                                         # prediction is a legitimate emergence
    dt: float = 0.1


class Tracker:
    def __init__(self, config: TrackerConfig | None = None, occluders=()):
        self.config = config or TrackerConfig()
        self.occluders: list[Occluder] = list(occluders)
        self.tracks: list[Track] = []
        self.frame = 0
        self.convergence_records: list[ConvergenceRecord] = []
        self.loss_records: list[LossRecord] = []

    # ==========================================================================
    # Frame update
    # ==========================================================================
    def step(self, detections, *, dt: float | None = None) -> list[TrackFrameReport]:
        self.frame += 1
        dt = self.config.dt if dt is None else dt
        detections = list(detections)

        claimed = [False] * len(detections)
        survivors: list[Track] = []
        for track in self.tracks:
            if self._update_track(track, detections, claimed, dt):
                survivors.append(track)
        self.tracks = survivors

        # Whatever no surviving explanation accounts for may genuinely be new.
        for index, detection in enumerate(detections):
            if not claimed[index]:
                self._birth_track(detection)

        self._suppress_duplicate_tracks()
        return [self._build_report(track) for track in self.tracks]

    # ==========================================================================
    # Per-track branching
    # ==========================================================================
    def _update_track(self, track: Track, detections, claimed, dt: float) -> bool:
        """Returns False when the track is deleted this frame."""
        cfg = self.config

        predicted: list[tuple[Hypothesis, np.ndarray, np.ndarray]] = []
        for hyp in track.hypotheses:
            px, pP = predict_hypothesis(hyp.x, hyp.P, dt, cfg.q_sigma)
            predicted.append((hyp, px, pP))

        # Gate against the strongest prediction; each gated detection still gets
        # its own branch, so every locally plausible association stays alive.
        reference = max(predicted, key=lambda item: item[0].log_weight)
        gate_info = self._gate(reference[1], reference[2], detections)

        children: list[Hypothesis] = []
        for parent, px, pP in predicted:
            predicted_covered = self._covered(px)
            # ---- measurement branches ---------------------------------------
            for index, d2 in gate_info:
                detection = detections[index]
                # A sighting is only a legal association if the target could
                # physically be seen at that location. While it is predicted to
                # be behind an occluder, a detection inside the same occluder
                # keeps the low residual visibility, and a detection reachable
                # without crossing the occluder's boundary is accepted as a
                # genuine emergence. A sighting on the far side (line of sight
                # crosses the occluder) is rejected outright — admitting it even
                # once creates a self-reinforcing branch that drags the track
                # onto a distractor for all later frames.
                sighting_covered = self._covered(detection.z())
                if predicted_covered and not sighting_covered:
                    # A legitimate emergence hugs the occluder's *boundary*
                    # (the target stepping out from behind it). Judging by
                    # distance to the predicted mean would be wrong: late in a
                    # long occlusion the covariance is large and the mean can
                    # sit beyond the far edge, which would admit a far-away
                    # distractor as a "nearby emergence".
                    near_edge = self._distance_to_occluder(detection.z()) <= cfg.emerge_distance
                    if not near_edge and not self._line_of_sight_clear(px[:2], detection.z()):
                        continue
                    p_d = cfg.p_detect
                elif predicted_covered:
                    p_d = cfg.p_detect_occluded
                else:
                    p_d = cfg.p_detect
                x_new, P_new, *_ = update_hypothesis(
                    px, pP, detection, cfg.measurement_variance)
                children.append(parent.clone(
                    x=x_new, P=P_new,
                    log_weight=parent.log_weight + math.log(max(p_d, 1e-12))
                    + self._association_log_weight(d2),
                    tag=self._lineage_tag(parent, x_new),
                    last_measurement_frame=self.frame,
                    hits=parent.hits + 1, misses=0,
                    last_mahalanobis=d2, origin='measurement',
                    last_detection=(float(detection.x), float(detection.y)),
                    last_detection_index=index))
            # ---- miss / coast branch (always exists) ------------------------
            p_miss = (1.0 - cfg.p_detect_occluded) if predicted_covered else (1.0 - cfg.p_detect)
            children.append(parent.clone(
                x=px, P=pP,
                log_weight=parent.log_weight + math.log(max(p_miss, 1e-9)),
                misses=parent.misses + 1, hits=0, origin='coast',
                last_mahalanobis=None, last_detection_index=None))

        children = self._normalise(children)
        children = self._merge_branches(children)
        children = self._prune(children)
        track.hypotheses = children

        best = track.best()
        if best.origin == 'measurement' and best.last_detection_index is not None:
            claimed[best.last_detection_index] = True

        return self._advance_status(track, best, detections, gate_info)

    # ------------------------------------------------------------------ gating
    def _gate(self, px: np.ndarray, pP: np.ndarray, detections):
        """All detections inside the chi-square gate with (index, d^2)."""
        H = _OBSERVATION
        out = []
        for index, detection in enumerate(detections):
            R_d = measurement_covariance(detection, self.config.measurement_variance)
            S = H @ pP @ H.T + R_d
            innovation = detection.z() - H @ px
            d2 = float(innovation @ np.linalg.solve(S, innovation))
            if math.isfinite(d2) and d2 <= self.config.gate:
                out.append((index, d2))
        return out

    def _association_log_weight(self, d2: float) -> float:
        """Likelihood ratio of "target generated this detection" against the
        uniform clutter model: (1/lambda) * exp(-d^2/2). Using the kernel form
        (instead of the fully normalised density) keeps an association preferred
        to a miss even when the coasted innovation covariance has grown large —
        otherwise the 1/|S| normalising term rewards staying hidden forever."""
        return -math.log(max(self.config.clutter_density, 1e-12)) - 0.5 * d2

    @staticmethod
    def _nearest_mahalanobis(px: np.ndarray, pP: np.ndarray, detections,
                             default_variance: float) -> tuple[int, float] | None:
        if not detections:
            return None
        H = _OBSERVATION
        best_index, best_d2 = -1, float('inf')
        for index, detection in enumerate(detections):
            R_d = measurement_covariance(detection, default_variance)
            S = H @ pP @ H.T + R_d
            innovation = detection.z() - H @ px
            d2 = float(innovation @ np.linalg.solve(S, innovation))
            if d2 < best_d2:
                best_index, best_d2 = index, d2
        return (best_index, best_d2) if best_index >= 0 else None

    # ------------------------------------------------------------- hypothesis
    def _lineage_tag(self, parent: Hypothesis, x_new: np.ndarray) -> int:
        """Measurement children inherit the parent's interpretation while the
        association stays spatially consistent; a jump beyond ``tag_radius``
        opens a *new* explanation (reconnecting to a different cluster)."""
        if parent.origin != 'measurement' or parent.last_detection is None:
            return parent.tag
        gap = math.dist(x_new[:2], parent.last_detection)
        return parent.tag if gap <= self.config.tag_radius else next(TAG_IDS)

    @staticmethod
    def _normalise(children: list[Hypothesis]) -> list[Hypothesis]:
        log_w = np.array([h.log_weight for h in children], dtype=float)
        total = float(np.logaddexp.reduce(log_w))
        for hyp in children:
            hyp.log_weight -= total
        return children

    def _merge_branches(self, children: list[Hypothesis]) -> list[Hypothesis]:
        """Collapse same-tag, spatially-coincident branches (different parents
        often branch to the same detection) into one Gaussian mixture."""
        groups: list[list[Hypothesis]] = []
        for hyp in sorted(children, key=lambda h: h.log_weight, reverse=True):
            for group in groups:
                head = group[0]
                if head.tag == hyp.tag and math.dist(head.x[:2], hyp.x[:2]) <= self.config.tag_radius:
                    group.append(hyp)
                    break
            else:
                groups.append([hyp])

        merged: list[Hypothesis] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            weights = np.array([math.exp(h.log_weight) for h in group], dtype=float)
            weights /= weights.sum()
            mean = np.zeros_like(group[0].x)
            for w, h in zip(weights, group):
                mean += w * h.x
            covariance = np.zeros_like(group[0].P)
            for w, h in zip(weights, group):
                delta = (h.x - mean).reshape(-1, 1)
                covariance += w * (h.P + delta @ delta.T)
            head = max(group, key=lambda h: h.log_weight)
            merged.append(head.clone(
                x=mean, P=(covariance + covariance.T) / 2.0,
                log_weight=float(np.logaddexp.reduce([h.log_weight for h in group])),
                hits=max(h.hits for h in group),
                misses=min(h.misses for h in group),
                last_measurement_frame=max(h.last_measurement_frame for h in group),
                last_detection=head.last_detection,
                last_detection_index=head.last_detection_index,
                last_mahalanobis=head.last_mahalanobis,
                origin=head.origin))
        return merged

    def _prune(self, children: list[Hypothesis]) -> list[Hypothesis]:
        kept = [h for h in children if math.exp(h.log_weight) >= self.config.prune_weight]
        kept.sort(key=lambda h: h.log_weight, reverse=True)
        kept = kept[: self.config.max_hypotheses]
        if not kept:
            kept = [max(children, key=lambda h: h.log_weight)]
        return self._normalise(kept)

    # ==========================================================================
    # Births and status machine
    # ==========================================================================
    def _birth_track(self, detection: Detection) -> Track:
        R = measurement_covariance(detection, self.config.measurement_variance)
        x = np.array([detection.x, detection.y, 0.0, 0.0], dtype=float)
        P = np.diag([max(float(R[0, 0]), 1e-3), max(float(R[1, 1]), 1e-3), 4.0, 4.0])
        hyp = Hypothesis(x=x, P=P, log_weight=0.0, tag=next(TAG_IDS),
                         birth_frame=self.frame, last_measurement_frame=self.frame,
                         hits=1, origin='birth',
                         last_detection=(float(detection.x), float(detection.y)))
        track = Track(created_frame=self.frame, hypotheses=[hyp],
                      status=TrackStatus.TENTATIVE, best_tag=hyp.tag, hit_streak=1)
        self.tracks.append(track)
        return track

    def _advance_status(self, track: Track, best: Hypothesis, detections,
                        gate_info) -> bool:
        """Update lifecycle; returns False when the track should be deleted."""
        cfg = self.config
        observed = best.origin == 'measurement'
        covered, label = self._covered_with_label(best.x)

        if observed:
            track.miss_streak = 0
            track.unexplained_misses = 0
            track.hit_streak += 1
            track.last_occluder = None
            track.last_covered_frame = None
        else:
            track.miss_streak += 1
            track.hit_streak = 0
            if covered:
                # An occluder fully explains the miss: do not age the track for
                # "lost" purposes while the prediction remains covered.
                track.unexplained_misses = 0
                track.last_covered_frame = self.frame
                track.last_occluder = label
            else:
                recent_cover = (track.last_covered_frame is not None
                                and self.frame - track.last_covered_frame <= cfg.reacquire_grace)
                track.unexplained_misses = (track.unexplained_misses + 1
                                            if not recent_cover else track.unexplained_misses)

        # Dominance tracking drives deferred commitment.
        if best.tag == track.best_tag:
            track.dominance_streak += 1
        else:
            track.best_tag = best.tag
            track.dominance_streak = 1

        weights = track.weights()
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        best_w = float(weights[order[0]])
        second_w = float(weights[order[1]]) if len(order) > 1 else 0.0
        ratio = second_w / best_w if best_w > 0 else 0.0
        nearest = self._nearest_mahalanobis(best.x, best.P, detections,
                                            cfg.measurement_variance)
        prior = track.status
        reacquired = False

        if prior == TrackStatus.TENTATIVE:
            if observed and track.hit_streak >= cfg.confirm_hits:
                track.status = TrackStatus.CONFIRMED
                track.committed_tag = best.tag
                track.events.append(f'frame {self.frame}: confirmed after '
                                    f'{track.hit_streak} corroborating hits')
                self._maybe_converge(track, best, best_w, ratio, detections,
                                     gate_info, reacquired=False, newly_confirmed=True)
                return True
            if track.miss_streak > cfg.tentative_max_misses:
                self._declare_loss(
                    track, TrackStatus.TENTATIVE, reason='tentative_unsupported',
                    detail='Uncorroborated birth never received enough hits to be '
                           'confirmed; interpreted as clutter and removed.',
                    nearest=nearest, occluded=covered, label=label)
                return False
            self._maybe_converge(track, best, best_w, ratio, detections, gate_info)
            return True

        if prior in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED):
            if observed:
                reacquired = prior == TrackStatus.OCCLUDED or track.coast_start_frame is not None
                if reacquired:
                    gap = (self.frame - track.coast_start_frame
                           if track.coast_start_frame is not None else 1)
                    track.events.append(
                        f'frame {self.frame}: reacquired after {gap} frames coasting')
                track.status = TrackStatus.CONFIRMED
                track.coast_start_frame = None
            elif covered:
                if track.coast_start_frame is None:
                    track.coast_start_frame = self.frame
                track.status = TrackStatus.OCCLUDED
                coasted = self.frame - track.coast_start_frame
                if coasted > cfg.occlusion_budget_frames:
                    self._declare_loss(
                        track, TrackStatus.OCCLUDED, reason='occlusion_timeout',
                        detail=f'Predicted position stayed inside "{label}" for '
                               f'{coasted} frames, past the '
                               f'{cfg.occlusion_budget_frames}-frame occlusion budget; '
                               f'coasting on stale velocity no longer explains any detection.',
                        nearest=nearest, occluded=True, label=label)
                    track.status = TrackStatus.LOST
                    track.last_loss_frame = self.frame
                    track.coast_start_frame = None
                elif prior == TrackStatus.CONFIRMED:
                    track.events.append(
                        f'frame {self.frame}: occluded by "{label}", coasting')
            else:
                if track.unexplained_misses >= cfg.lost_after_misses:
                    if prior == TrackStatus.OCCLUDED:
                        track.events.append(
                            f'frame {self.frame}: cleared occluder without reappearing')
                    self._declare_loss(
                        track, prior, reason='no_measurement_no_occluder',
                        detail=f'{track.unexplained_misses} consecutive misses are not '
                               'explained by any known occluder and no detection falls '
                               'inside the association gate; the target likely left the '
                               'field of view or the detector stopped firing. Coasted '
                               'briefly for reacquisition.',
                        nearest=nearest, occluded=False, label=None)
                    track.status = TrackStatus.LOST
                    track.last_loss_frame = self.frame
                    track.coast_start_frame = None
                else:
                    # Still inside the post-occlusion reacquisition grace: keep
                    # the OCCLUDED label (it really just emerged) but wait.
                    track.status = TrackStatus.OCCLUDED if prior == TrackStatus.OCCLUDED else track.status

        elif prior == TrackStatus.LOST:
            if observed:
                reacquired = True
                track.status = TrackStatus.CONFIRMED
                track.miss_streak = 0
                track.committed_tag = best.tag
                track.events.append(
                    f'frame {self.frame}: late reacquisition inside the coast window')
            elif self._frames_since_loss(track) > cfg.lost_coast_frames:
                self._declare_loss(
                    track, TrackStatus.LOST, reason='coast_window_expired',
                    detail=f'No reacquisition within {cfg.lost_coast_frames} frames '
                           'after the loss; the track is deleted.',
                    nearest=nearest, occluded=covered, label=label,
                    budget=cfg.lost_coast_frames)
                return False

        self._maybe_converge(track, best, best_w, ratio, detections, gate_info,
                             reacquired=reacquired)
        return True

    def _maybe_converge(self, track, best, best_w, ratio, detections,
                        gate_info, *, reacquired: bool = False,
                        newly_confirmed: bool = False) -> None:
        cfg = self.config
        competing = len({h.tag for h in track.hypotheses}) > 1
        if best_w < cfg.converge_weight or ratio > cfg.ambiguity_ratio:
            return
        if not (competing or reacquired or newly_confirmed):
            return
        if track.dominance_streak < cfg.converge_frames:
            return
        if track.committed_tag == best.tag and not reacquired:
            return

        candidates = tuple(self._candidate_records(track, gate_info, detections))
        evidence: list[str] = []
        if reacquired:
            evidence.append('reacquired by a measurement branch that beat the coast/miss branch')
        if competing:
            evidence.append(
                f'{len({h.tag for h in track.hypotheses})} interpretations competed; '
                f'winner posterior {best_w:.3f}, second/first ratio {ratio:.3f} '
                f'(< {cfg.ambiguity_ratio})')
        evidence.append(f'winning interpretation sustained for {track.dominance_streak} frames')
        if best.last_mahalanobis is not None:
            evidence.append(f'winning match at Mahalanobis d^2={best.last_mahalanobis:.3f} '
                            f'inside gate {cfg.gate:.3f}')

        record = ConvergenceRecord(
            frame=self.frame, chosen_tag=best.tag, winning_weight=best_w,
            support_frames=track.dominance_streak,
            margin=best_w * (1.0 - ratio),
            ambiguity_ratio=ratio, evidence=tuple(evidence), candidates=candidates)
        self.convergence_records.append(record)
        track.committed_tag = best.tag
        track.events.append(f'frame {self.frame}: converged on tag {best.tag} '
                            f'(w={best_w:.3f}, ratio={ratio:.3f})')
        track.pending = {'convergence': record}

    def _candidate_records(self, track, gate_info, detections) -> list[AssociationRecord]:
        weight_by_index: dict[int, float] = {}
        for hyp in track.hypotheses:
            if hyp.origin == 'measurement' and hyp.last_detection_index is not None:
                weight_by_index[hyp.last_detection_index] = max(
                    weight_by_index.get(hyp.last_detection_index, 0.0),
                    math.exp(hyp.log_weight))
        records: list[AssociationRecord] = []
        for index, d2 in gate_info:
            records.append(AssociationRecord(
                detection_index=index, mahalanobis=d2, gate=self.config.gate,
                inside_gate=True, log_likelihood=self._association_log_weight(d2),
                weight_after=weight_by_index.get(index, 0.0)))
        if not records:
            nearest = self._nearest_mahalanobis(track.best().x, track.best().P,
                                                detections, self.config.measurement_variance)
            if nearest is not None:
                index, d2 = nearest
                records.append(AssociationRecord(
                    detection_index=index, mahalanobis=d2, gate=self.config.gate,
                    inside_gate=False, log_likelihood=float('-inf'), weight_after=0.0))
        return sorted(records, key=lambda r: r.weight_after, reverse=True)[:4]

    def _suppress_duplicate_tracks(self) -> None:
        """Merge tracks that describe the *same* object.

        Two tracks are duplicates only when the evidence is unambiguous: their
        best estimates are within ``merge_distance`` *and* either one side is a
        tentative newcomer, or both are confirmed and currently latch onto the
        same detection. Two tracks near each other but attached to different
        detections (a genuine crossing) are left alone so their explanations
        keep competing.
        """
        absorbed: set[int] = set()
        tracks = [t for t in self.tracks if t.id not in absorbed]
        for i, a in enumerate(tracks):
            if a.id in absorbed:
                continue
            for b in tracks[i + 1:]:
                if b.id in absorbed:
                    continue
                if not self._are_duplicates(a, b):
                    continue
                winner, loser = self._rank_duplicate(a, b)
                absorbed.add(loser.id)
                loser_w = float(max(math.exp(h.log_weight) for h in loser.hypotheses))
                winner.events.append(
                    f'frame {self.frame}: absorbed duplicate track {loser.id} '
                    f'({loser.status.value}; both explain the same object, '
                    f'absorbed weight {loser_w:.3f})')
        self.tracks = [t for t in self.tracks if t.id not in absorbed]

    def _are_duplicates(self, a: Track, b: Track) -> bool:
        if math.dist(a.best().x[:2], b.best().x[:2]) > self.config.merge_distance:
            return False
        a_best, b_best = a.best(), b.best()
        if TrackStatus.TENTATIVE in (a.status, b.status):
            # A tentative detection that an established track already explains
            # is clutter / a double report of the same object.
            other = b if a.status == TrackStatus.TENTATIVE else a
            return other.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
        both_firm = all(t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
                        for t in (a, b))
        if not both_firm:
            return False
        # Two established tracks that latch onto the *same* detection and have
        # been doing so for a couple of frames are the same object after a
        # crossing; a single-frame meeting on different detections is not merged.
        common_detection = (a_best.origin == 'measurement'
                            and b_best.origin == 'measurement'
                            and a_best.last_detection_index is not None
                            and a_best.last_detection_index == b_best.last_detection_index)
        sustained = min(a.best().hits, b.best().hits) >= 2
        return common_detection and sustained

    @staticmethod
    def _rank_duplicate(a: Track, b: Track):
        def score(t: Track) -> tuple:
            # Prefer confirmed, then older, then better hit history/weight.
            return (t.status == TrackStatus.CONFIRMED,
                    t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED),
                    -t.created_frame, t.best().hits, -t.miss_streak,
                    t.best().log_weight)
        return (a, b) if score(a) >= score(b) else (b, a)

    # ==========================================================================
    # Occlusion / loss helpers
    # ==========================================================================
    def _covered(self, x: np.ndarray) -> bool:
        return self._covered_with_label(x)[0]

    def _covered_with_label(self, x: np.ndarray) -> tuple[bool, str | None]:
        for occluder in self.occluders:
            if occluder.covers(float(x[0]), float(x[1])):
                return True, occluder.label
        return False, None

    def _line_of_sight_clear(self, start, end) -> bool:
        """False when any occluder's interior lies between the two points."""
        for occluder in self.occluders:
            if occluder.blocks(start, end):
                return False
        return True

    def _distance_to_occluder(self, point) -> float:
        """Minimum distance from an outside point to any occluder boundary."""
        best = float('inf')
        for occluder in self.occluders:
            vertices = list(occluder.polygon)
            for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                best = min(best, _point_segment_distance(point, a, b))
        return best

    def _declare_loss(self, track: Track, status: TrackStatus, *, reason: str,
                      detail: str, nearest, occluded: bool,
                      label: str | None, budget: int | None = None) -> None:
        best = track.best()
        nearest_d2 = None
        if nearest is not None:
            _, nearest_d2 = nearest
        if budget is None:
            budget = (self.config.occlusion_budget_frames if occluded
                      else self.config.lost_after_misses)
        record = LossRecord(
            track_id=track.id, frame=self.frame, status=status, reason=reason,
            consecutive_misses=track.miss_streak,
            miss_budget=budget,
            occluded=occluded, occluder_label=label,
            predicted_position=(float(best.x[0]), float(best.x[1])),
            last_observed_position=best.last_detection,
            last_observed_frame=(best.last_measurement_frame
                                 if best.last_measurement_frame >= 0 else None),
            nearest_detection_mahalanobis=nearest_d2, detail=detail)
        self.loss_records.append(record)
        track.events.append(f'frame {self.frame}: {reason}')

    def _frames_since_loss(self, track: Track) -> int:
        if track.last_loss_frame is None:
            return 0
        return self.frame - track.last_loss_frame

    # ==========================================================================
    # Reporting
    # ==========================================================================
    def _build_report(self, track: Track) -> TrackFrameReport:
        best = track.best()
        weights = track.weights()
        ordered = sorted((float(w) for w in weights), reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        ambiguity = second / ordered[0] if ordered[0] > 0 else 0.0
        pending_conv = None
        if isinstance(track.pending, dict):
            pending_conv = track.pending.get('convergence')
        fresh_loss = next((r for r in reversed(self.loss_records)
                           if r.track_id == track.id and r.frame == self.frame), None)
        return TrackFrameReport(
            track_id=track.id, status=track.status,
            position=(float(best.x[0]), float(best.x[1])),
            velocity=(float(best.x[2]), float(best.x[3])),
            best_weight=float(weights.max()), hypotheses=len(track.hypotheses),
            distinct_interpretations=len({h.tag for h in track.hypotheses}),
            consecutive_misses=track.miss_streak, ambiguity=ambiguity,
            convergence=pending_conv if (pending_conv is not None
                                         and pending_conv.frame == self.frame) else None,
            loss=fresh_loss, notes=list(track.events)[-4:])

    # ==========================================================================
    # Introspection
    # ==========================================================================
    def competing_explanations(self, track_id: int) -> list[dict]:
        """Every live interpretation of one track with its supporting evidence."""
        track = next((t for t in self.tracks if t.id == track_id), None)
        if track is None:
            return []
        weights = track.weights()
        out = []
        for hyp, weight in sorted(zip(track.hypotheses, weights),
                                  key=lambda pair: pair[1], reverse=True):
            out.append({
                'tag': hyp.tag, 'weight': float(weight),
                'position': (float(hyp.x[0]), float(hyp.x[1])),
                'velocity': (float(hyp.x[2]), float(hyp.x[3])),
                'origin': hyp.origin, 'hits': hyp.hits, 'misses': hyp.misses,
                'mahalanobis': hyp.last_mahalanobis,
                'last_detection_index': hyp.last_detection_index,
                'last_measurement_frame': hyp.last_measurement_frame})
        return out

    def explain_loss(self, track_id: int) -> LossRecord | None:
        return next((r for r in reversed(self.loss_records)
                     if r.track_id == track_id), None)
