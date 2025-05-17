# 오차 벡터 계산
import numpy as np
from typing import Callable

def compute_error_vector(p: np.ndarray, r_func: Callable, t: float) -> np.ndarray:
    r_t = r_func(t)
    return p - r_t