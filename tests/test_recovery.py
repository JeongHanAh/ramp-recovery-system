# 단위 테스트 예시
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recovery_algo import compute_correction_vector, update_position

def test_compute_correction_vector():
    r_dot = np.array([1.0, 0.0])
    delta = np.array([0.0, 1.0])
    alpha, beta = 0.5, 0.5
    B = compute_correction_vector(r_dot, delta, alpha, beta)
    assert np.allclose(B, np.array([0.5, 0.5]))

def test_update_position():
    p = np.array([1.0, 1.0])
    B = np.array([0.1, 0.2])
    eta = 1.0
    delta_t = 1.0
    new_p = update_position(p, B, eta, delta_t)
    assert np.allclose(new_p, np.array([1.1, 1.2]))