import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from dataclasses import dataclass
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.ramp_geometry import RampGeometry

@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray
    gps_position: Optional[np.ndarray] = None

@dataclass
class GPSMeasurement:
    position: np.ndarray  # [x, y, z]
    timestamp: float
    accuracy: float  # 추정된 정확도 (미터)

class GPSDistortionSimulator:
    def __init__(self, noise_std: float = 2.0, bias_scale: float = 1.0):
        """GPS 왜곡 시뮬레이터 초기화
        Args:
            noise_std: GPS 노이즈의 표준편차 (미터)
            bias_scale: 시스템적 편향의 크기 (미터)
        """
        self.noise_std = noise_std
        self.bias_scale = bias_scale
        self.bias = np.random.normal(0, bias_scale, 3)  # 3D 바이어스
        
    def apply_distortion(self, true_position: np.ndarray, elevation: float) -> np.ndarray:
        """실제 위치에 GPS 왜곡 적용"""
        # 3D 노이즈 생성 (수직 방향 노이즈는 더 크게)
        noise = np.random.normal(0, self.noise_std, 3)
        noise[2] *= 1.5  # 수직 방향 노이즈 증가
        
        # 바이어스 천천히 업데이트
        self.bias += np.random.normal(0, 0.1, 3)
        self.bias = np.clip(self.bias, -self.bias_scale*2, self.bias_scale*2)
        
        return true_position + noise + self.bias

class GPSCorrection:
    def __init__(self, ramp_geometry: RampGeometry):
        """GPS 보정 시스템 초기화
        Args:
            ramp_geometry: 램프 형상 정보
        """
        self.ramp = ramp_geometry
        self.alpha = 0.7  # 보정 강도
        self.beta = 0.3   # 속도 영향 가중치
        
    def correct_position(self, 
                        vehicle_state: VehicleState,
                        dt: float,
                        ramp_index: int = 0) -> VehicleState:
        """GPS 위치 보정
        Args:
            vehicle_state: 현재 차량 상태
            dt: 시간 간격
            ramp_index: 현재 주행 중인 램프 인덱스
        Returns:
            보정된 차량 상태
        """
        # 1. GPS 위치를 곡선에 투영
        t, projected_point = self.ramp.project_point(vehicle_state.gps_position, ramp_index)
        
        # 2. 곡선의 기하학적 특성 계산
        tangent = self.ramp.get_tangent(t, ramp_index)
        normal = np.array([-tangent[1], tangent[0]])
        curvature = self.ramp.get_curvature(t, ramp_index)
        
        # 3. 속도 기반 가중치 계산
        speed = np.linalg.norm(vehicle_state.velocity)
        velocity_weight = np.exp(-self.beta * speed)
        
        # 4. 오차 분해
        error = vehicle_state.gps_position - projected_point
        tangential_error = np.dot(error, tangent)
        normal_error = np.dot(error, normal)
        
        # 5. 보정 벡터 계산
        # 곡률이 큰 구간에서는 법선 방향 보정 강화
        curvature_weight = 1.0 / (1.0 + abs(curvature))
        correction = (
            -tangential_error * tangent * velocity_weight * self.alpha +
            -normal_error * normal * (1 + (1-curvature_weight)) * self.alpha
        )
        
        # 6. 보정된 위치 계산
        corrected_position = vehicle_state.position + correction
        
        # 7. 속도 업데이트 (보정 방향을 고려)
        new_velocity = (corrected_position - vehicle_state.position) / dt
        
        return VehicleState(
            position=corrected_position,
            velocity=new_velocity,
            gps_position=vehicle_state.gps_position
        ) 

class GPSCorrector:
    def __init__(self, window_size: int = 5, max_acceleration: float = 1.0, ramp_constraint_weight: float = 1.5):
        """GPS 보정기 초기화
        Args:
            window_size: 이동 평균 윈도우 크기
            max_acceleration: 최대 허용 가속도 (m/s^2)
            ramp_constraint_weight: 램프 구조 제약 가중치
        """
        self.window_size = window_size
        self.max_acceleration = max_acceleration
        self.ramp_constraint_weight = ramp_constraint_weight
        self.measurements = []
        self.last_position = None
        self.last_velocity = np.zeros(3)
        self.dt = 0.1  # 시간 간격
        
        # 칼만 필터 상태 초기화
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 10.0  # 초기 불확실성
        self.Q = np.eye(6) * 0.1   # 프로세스 노이즈
        self.R = np.eye(3) * 2.0   # 측정 노이즈
        
    def kalman_predict(self, dt: float):
        """칼만 필터 예측 단계"""
        # 상태 전이 행렬
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        
        # 상태 예측
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
    def kalman_update(self, measurement: np.ndarray):
        """칼만 필터 업데이트 단계"""
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # 칼만 게인 계산
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        innovation = measurement - H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
            
    def add_measurement(self, measurement: GPSMeasurement):
        """새로운 GPS 측정값 추가"""
        self.measurements.append(measurement)
        if len(self.measurements) > self.window_size:
            self.measurements.pop(0)
            
    def correct_position(self, ramp_distance_func) -> np.ndarray:
        """GPS 위치 보정
        Args:
            ramp_distance_func: 램프까지의 거리를 계산하는 함수
        Returns:
            보정된 3D 위치
        """
        if not self.measurements:
            return None
            
        current_measurement = self.measurements[-1]
        current_position = current_measurement.position
        
        # 1. 칼만 필터 예측
        self.kalman_predict(self.dt)
        
        # 2. 이동 평균으로 노이즈 감소 (적응형 윈도우)
        window_size = min(len(self.measurements), self.window_size)
        if window_size >= 2:
            recent_positions = np.array([m.position for m in self.measurements[-window_size:]])
            position_std = np.std(recent_positions, axis=0)
            
            # 노이즈 수준에 따른 적응형 가중치
            noise_weight = np.exp(-np.linalg.norm(position_std))
            smoothed_position = np.average(recent_positions, 
                                         weights=np.exp(-np.arange(window_size)[::-1]),
                                         axis=0)
            
            # 급격한 변화 방지 (적응형 제한)
            max_change = 3.0 * (1.0 + noise_weight)  # 노이즈가 적을 때 더 큰 변화 허용
            diff = smoothed_position - current_position
            if np.linalg.norm(diff) > max_change:
                diff = diff * max_change / np.linalg.norm(diff)
            current_position = current_position + diff * (0.7 + 0.2 * noise_weight)
        
        # 3. 램프 구조 제약 적용 (적응형 보정)
        ramp_distance = ramp_distance_func(current_position)
        _, closest_point = ramp_distance_func(current_position, return_closest_point=True)
        
        # 거리 및 속도 기반 적응형 보정 강도
        if self.last_position is not None:
            velocity = (current_position - self.last_position) / self.dt
            speed = np.linalg.norm(velocity)
            speed_factor = np.exp(-speed / 5.0)  # 고속에서는 보정 강도 감소
        else:
            speed_factor = 1.0
            
        base_strength = np.clip(ramp_distance / 3.0, 0, 0.95)
        correction_strength = base_strength * (0.8 + 0.2 * speed_factor)
        
        # 비선형 보정 적용 (거리에 따른 지수적 감소)
        distance_weight = np.exp(-ramp_distance / 2.0)
        corrected_position = (current_position * (1 - correction_strength) + 
                            closest_point * correction_strength) * (1 - distance_weight) + \
                           closest_point * distance_weight
        
        # 4. 수직 방향 적응형 보정
        if self.last_position is not None:
            vertical_change = corrected_position[2] - self.last_position[2]
            max_vertical_change = 1.0 * (1.0 + speed_factor)  # 저속에서 더 큰 수직 변화 허용
            if abs(vertical_change) > max_vertical_change:
                corrected_position[2] = self.last_position[2] + np.sign(vertical_change) * max_vertical_change
        
        # 5. 칼만 필터 업데이트
        self.kalman_update(corrected_position)
        final_position = self.state[:3]  # 칼만 필터 상태에서 위치 추출
        
        # 최종 위치는 보정된 위치와 칼만 필터 추정치의 가중 평균
        kalman_weight = 0.3
        final_position = corrected_position * (1 - kalman_weight) + final_position * kalman_weight
        
        self.last_position = final_position.copy()
        return final_position
        
    def get_correction_uncertainty(self) -> float:
        """현재 보정의 불확실성 추정
        Returns:
            추정된 불확실성 (미터)
        """
        if len(self.measurements) < 2:
            return float('inf')
        positions = np.array([m.position for m in self.measurements])
        return float(np.std(positions, axis=0).mean()) 