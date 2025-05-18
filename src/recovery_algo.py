# 보정 벡터 계산 및 위치 업데이트
import numpy as np
from typing import Callable, Tuple

from spline_curve import compute_derivative
from delta_vector import compute_error_vector

class VehicleState:
    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2)):
        """
        차량 상태 초기화
        Args:
            position: 현재 위치 [x, y]
            velocity: 현재 속도 [vx, vy]
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.gps_position = position.copy()  # GPS로 측정된 위치
        
    def update(self, dt: float, acceleration: np.ndarray):
        """
        차량 상태 업데이트
        Args:
            dt: 시간 간격
            acceleration: 가속도 [ax, ay]
        """
        # 실제 차량의 물리적 움직임 업데이트
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

def compute_correction_vector(vehicle_state: VehicleState, 
                            reference_path: Callable,
                            t: float,
                            params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    보정 벡터 계산
    Args:
        vehicle_state: 현재 차량 상태
        reference_path: 기준 경로 함수
        t: 현재 시간
        params: 알고리즘 파라미터
    Returns:
        (가속도 벡터, 보정된 GPS 위치)
    """
    # 기준 경로 상의 목표 지점과 방향
    target_position = reference_path(t)
    path_direction = compute_derivative(reference_path, t)
    path_direction = path_direction / np.linalg.norm(path_direction)
    
    # GPS 오차 보정
    gps_error = vehicle_state.gps_position - vehicle_state.position
    corrected_gps = vehicle_state.position + params['gps_trust'] * gps_error
    
    # 목표 지점까지의 방향과 거리
    to_target = target_position - vehicle_state.position
    distance = np.linalg.norm(to_target)
    
    if distance < 1e-6:
        return np.zeros(2), corrected_gps
    
    direction = to_target / distance
    
    # 경로 추종을 위한 가속도 계산
    acceleration = (params['path_weight'] * direction + 
                   params['tangent_weight'] * path_direction -
                   params['damping'] * vehicle_state.velocity)
    
    return acceleration, corrected_gps

def recovery_step(vehicle_state: VehicleState,
                 reference_path: Callable,
                 t: float,
                 dt: float = 0.1) -> Tuple[VehicleState, np.ndarray]:
    """
    한 스텝의 경로 복구 수행
    Args:
        vehicle_state: 현재 차량 상태
        reference_path: 기준 경로 함수
        t: 현재 시간
        dt: 시간 간격
    Returns:
        (업데이트된 차량 상태, 보정된 GPS 위치)
    """
    params = {
        'path_weight': 2.0,    # 경로 추종 가중치
        'tangent_weight': 1.0, # 접선 방향 가중치
        'damping': 0.5,        # 감쇠 계수
        'gps_trust': 0.3       # GPS 신뢰도
    }
    
    # 보정 벡터 계산
    acceleration, corrected_gps = compute_correction_vector(
        vehicle_state, reference_path, t, params)
    
    # 차량 상태 업데이트
    vehicle_state.update(dt, acceleration)
    
    return vehicle_state, corrected_gps