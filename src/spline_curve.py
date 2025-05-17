# 기준 경로 모델링 및 도함수 계산
from typing import Callable, List
import numpy as np
from scipy.interpolate import splrep, splev

def fit_spline(x: List[float], y: List[float]) -> Callable:
    tck_x = splrep(range(len(x)), x, s=0)
    tck_y = splrep(range(len(y)), y, s=0)
    def r_func(t: float) -> np.ndarray:
        return np.array([splev(t, tck_x), splev(t, tck_y)])
    return r_func

def compute_derivative(r_func: Callable, t: float, h: float = 1e-5) -> np.ndarray:
    return (r_func(t + h) - r_func(t - h)) / (2 * h)