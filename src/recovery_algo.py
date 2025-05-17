# 보정 벡터 계산 및 위치 업데이트
import numpy as np
from typing import Callable

from spline_curve import compute_derivative
from delta_vector import compute_error_vector

def compute_correction_vector(r_dot: np.ndarray, delta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha * r_dot + beta * delta

def update_position(p: np.ndarray, B: np.ndarray, eta: float, delta_t: float) -> np.ndarray:
    return p + eta * B * delta_t

def recovery_step(p: np.ndarray, r_func: Callable, t: float, alpha: float, beta: float, eta: float, delta_t: float) -> np.ndarray:
    r_dot = compute_derivative(r_func, t)
    delta = compute_error_vector(p, r_func, t)
    B = compute_correction_vector(r_dot, delta, alpha, beta)
    return update_position(p, B, eta, delta_t)