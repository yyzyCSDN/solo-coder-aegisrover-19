"""Constant-velocity motion model and Gaussian measurement likelihood.

State is 4-D ``[px, py, vx, vy]`` with a 2-D position measurement. Keeping the
linear algebra here (rather than inside the tracker) makes the branching rules
easy to read and test: every hypothesis is just a Gaussian mean/covariance with
an attached log-weight.
"""
from __future__ import annotations

import numpy as np

from .models import Detection

__all__ = ('STATE_DIM', 'transition', 'process_noise', 'measurement_matrix',
           'measurement_covariance', 'gaussian_log_likelihood', 'predict_hypothesis',
           'update_hypothesis')

STATE_DIM = 4
_MEASUREMENT = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]], dtype=float)


def transition(dt: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dt, 0.0],
                     [0.0, 1.0, 0.0, dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=float)


def process_noise(dt: float, sigma: float) -> np.ndarray:
    """Piecewise-constant white acceleration noise (Bar-Shalom form)."""
    q = np.array([[dt ** 3 / 3.0, 0.0, dt ** 2 / 2.0, 0.0],
                  [0.0, dt ** 3 / 3.0, 0.0, dt ** 2 / 2.0],
                  [dt ** 2 / 2.0, 0.0, dt, 0.0],
                  [0.0, dt ** 2 / 2.0, 0.0, dt]], dtype=float)
    return (sigma ** 2) * q


def measurement_matrix() -> np.ndarray:
    return _MEASUREMENT.copy()


def measurement_covariance(detection: Detection, default: float) -> np.ndarray:
    if detection.covariance is not None:
        return np.asarray(detection.covariance, dtype=float).reshape(2, 2)
    return np.eye(2) * default


def gaussian_log_likelihood(z: np.ndarray, hx: np.ndarray, S: np.ndarray) -> float:
    """Log N(z ; hx, S), including the normalising constants."""
    innovation = z - hx
    sign, log_det = np.linalg.slogdet(S)
    if sign <= 0:
        return float('-inf')
    solved = np.linalg.solve(S, innovation)
    return float(-0.5 * (innovation @ solved + log_det + 2.0 * np.log(2.0 * np.pi)))


def predict_hypothesis(x: np.ndarray, P: np.ndarray, dt: float, q_sigma: float):
    F = transition(dt)
    Q = process_noise(dt, q_sigma)
    return F @ x, _symmetrise(F @ P @ F.T + Q)


def update_hypothesis(x: np.ndarray, P: np.ndarray, detection: Detection,
                      default_R: float):
    """Kalman measurement update; returns (x', P', innovation, S, d2, log_lik)."""
    H = measurement_matrix()
    R = measurement_covariance(detection, default_R)
    z = detection.z()
    innovation = z - H @ x
    S = _symmetrise(H @ P @ H.T + R)
    d2 = float(innovation @ np.linalg.solve(S, innovation))
    K = P @ H.T @ np.linalg.inv(S)
    identity = np.eye(STATE_DIM)
    x_new = x + K @ innovation
    P_new = _symmetrise((identity - K @ H) @ P @ (identity - K @ H).T + K @ R @ K.T)
    log_lik = gaussian_log_likelihood(z, H @ x, S)
    return x_new, P_new, innovation, S, d2, log_lik


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0
