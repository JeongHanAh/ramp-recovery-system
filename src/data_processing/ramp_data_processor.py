import numpy as np
import json
from typing import Tuple, List, Dict, Callable
from scipy.signal import savgol_filter

class VehicleState:
    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2)):
        """차량 상태 초기화
        Args:
            position: 현재 위치 [x, y]
            velocity: 현재 속도 [vx, vy]
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.gps_position = position.copy()  # GPS로 측정된 위치
        
    def update(self, dt: float, acceleration: np.ndarray):
        """차량 상태 업데이트
        Args:
            dt: 시간 간격
            acceleration: 가속도 [ax, ay]
        """
        # 실제 차량의 물리적 움직임 업데이트
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class RampDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
    
    def load_data(self) -> None:
        """램프 데이터 로드"""
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
    
    def extract_ramp_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """램프 구간의 좌표 추출"""
        if self.raw_data is None:
            raise ValueError("데이터를 먼저 로드해주세요.")
        
        # 램프 구간 식별 (예: 곡률이 특정 임계값을 넘는 구간)
        x_coords = np.array(self.raw_data['coordinates']['x'])
        y_coords = np.array(self.raw_data['coordinates']['y'])
        
        # 곡률 계산
        dx = np.gradient(x_coords)
        dy = np.gradient(y_coords)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
        
        # 램프 구간 식별 (곡률이 높은 구간)
        ramp_mask = curvature > np.mean(curvature) + np.std(curvature)
        
        return x_coords[ramp_mask], y_coords[ramp_mask]
    
    def smooth_coordinates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """좌표 스무딩"""
        # 윈도우 크기를 데이터 길이의 약 1/4로 설정하고 홀수로 만듦
        window = min(len(x) // 4 * 2 + 1, 5)
        # 다항식 차수는 윈도우 크기보다 작아야 함
        poly_order = min(window - 1, 2)
        x_smooth = savgol_filter(x, window, poly_order)
        y_smooth = savgol_filter(y, window, poly_order)
        return x_smooth, y_smooth
    
    def process_data(self) -> Dict[str, np.ndarray]:
        """전체 데이터 처리 파이프라인"""
        self.load_data()
        x_ramp, y_ramp = self.extract_ramp_coordinates()
        x_smooth, y_smooth = self.smooth_coordinates(x_ramp, y_ramp)
        
        return {
            'x_raw': x_ramp,
            'y_raw': y_ramp,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth
        }

def create_bspline_path(control_points: np.ndarray, degree: int = 3):
    """B-스플라인 기반 경로 생성"""
    knots = generate_uniform_knots(len(control_points), degree)
    return lambda t: evaluate_bspline(t, control_points, knots, degree)

def compute_curvature_based_correction(position: np.ndarray, 
                                     velocity: np.ndarray,
                                     path_derivative: np.ndarray,
                                     path_second_derivative: np.ndarray) -> np.ndarray:
    """곡률 기반 보정 벡터 계산"""
    # 곡률 계산
    path_magnitude = np.linalg.norm(path_derivative)
    
    # 경로 미분값이 0에 가까운 경우 처리
    if path_magnitude < 1e-6:
        return np.zeros_like(position)
        
    numerator = (path_derivative[0] * path_second_derivative[1] - 
                path_derivative[1] * path_second_derivative[0])
    denominator = np.power(path_magnitude, 3)
    curvature = numerator / denominator
    
    # 법선 벡터 계산
    normal = np.array([-path_derivative[1], path_derivative[0]])
    normal_magnitude = np.linalg.norm(normal)
    
    # 법선 벡터가 0에 가까운 경우 처리
    if normal_magnitude < 1e-6:
        return np.zeros_like(position)
        
    normal = normal / normal_magnitude
    
    # 곡률 기반 보정력 계산
    correction_force = curvature * normal
    return correction_force

class KalmanFilter:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        self.state = initial_state
        self.covariance = initial_covariance
        
    def predict(self, dt: float):
        """상태 예측"""
        F = np.array([[1, dt], [0, 1]])  # 상태 전이 행렬
        self.state = F @ self.state
        Q = np.eye(2) * 0.1  # 프로세스 노이즈
        self.covariance = F @ self.covariance @ F.T + Q
        
    def update(self, measurement: np.ndarray):
        """측정값 기반 상태 업데이트"""
        H = np.eye(2)  # 측정 행렬
        R = np.eye(2) * 0.2  # 측정 노이즈
        
        K = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + R)
        self.state = self.state + K @ (measurement - H @ self.state)
        self.covariance = (np.eye(2) - K @ H) @ self.covariance 

def compute_improved_correction(position: np.ndarray,
                             velocity: np.ndarray,
                             path_derivative: np.ndarray,
                             path_second_derivative: np.ndarray,
                             gps_position: np.ndarray,
                             dt: float) -> np.ndarray:
    """개선된 GPS 왜곡 보정 알고리즘"""
    
    # 1. 경로 투영 계산
    path_magnitude = np.linalg.norm(path_derivative)
    if path_magnitude < 1e-6:
        return np.zeros_like(position)
    
    path_tangent = path_derivative / path_magnitude
    path_normal = np.array([-path_tangent[1], path_tangent[0]])
    
    # 2. GPS 오차 분해
    gps_error = gps_position - position
    tangential_error = np.dot(gps_error, path_tangent) * path_tangent
    normal_error = np.dot(gps_error, path_normal) * path_normal
    
    # 3. 곡률 기반 가중치 계산 (안정성 개선)
    try:
        curvature = np.abs(path_derivative[0] * path_second_derivative[1] - 
                          path_derivative[1] * path_second_derivative[0]) / \
                   np.power(path_magnitude, 3)
        curvature = np.clip(curvature, 0, 1.0)  # 곡률 범위 제한
    except:
        curvature = 0.0
    
    # 4. 오차 누적 방지를 위한 적응형 가중치 (개선)
    error_magnitude = np.linalg.norm(gps_error)
    error_threshold = 0.2  # 오차 임계값 증가
    error_weight = np.clip(error_magnitude / error_threshold, 0, 1)
    
    # 5. 속도 기반 보정 강도 조절 (개선)
    speed = np.linalg.norm(velocity)
    min_speed_weight = 0.3  # 최소 가중치 설정
    speed_weight = min_speed_weight + (1.0 - min_speed_weight) * np.exp(-speed / 5.0)
    
    # 6. 방향별 보정 가중치 계산 (개선)
    base_weight = 0.5  # 기본 가중치 증가
    tangential_weight = base_weight * speed_weight
    normal_weight = base_weight * (1.0 + np.exp(-curvature * 5.0)) * 0.5
    
    # 7. 보정 벡터 계산
    correction = (tangential_weight * tangential_error + 
                 normal_weight * normal_error)
    
    # 8. 시간 간격 기반 보정량 제한 (개선)
    max_step = 0.05  # 최대 스텝 크기 증가
    dt_factor = np.clip(dt / 0.1, 0.5, 2.0)  # 시간 간격에 따른 보정량 조절 개선
    max_correction = max_step * dt_factor
    
    correction_magnitude = np.linalg.norm(correction)
    if correction_magnitude > max_correction:
        correction = correction * (max_correction / correction_magnitude)
    
    # 9. 이력 효과를 고려한 감쇠 (개선)
    history_damping = 0.95  # 이력 효과 증가
    correction *= history_damping
    
    return correction

class ImprovedKalmanFilter:
    def __init__(self, initial_state: np.ndarray):
        """향상된 칼만 필터 초기화"""
        self.state = initial_state
        self.P = np.eye(4) * 0.01  # 초기 불확실성 감소
        self.prev_correction = np.zeros(2)  # 이전 보정값 저장
        
    def predict(self, dt: float, acceleration: np.ndarray):
        """비선형 시스템 모델 기반 예측"""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # 상태 예측
        self.state = F @ self.state + B @ acceleration
        
        # 적응형 프로세스 노이즈
        Q = np.eye(4)
        Q[:2, :2] *= 0.005  # 위치 노이즈 더욱 감소
        Q[2:, 2:] *= 0.02   # 속도 노이즈 감소
        
        self.P = F @ self.P @ F.T + Q
        
    def update(self, measurement: np.ndarray, R: np.ndarray = None):
        """적응형 측정 업데이트"""
        if R is None:
            R = np.eye(2) * 0.1  # 기본 측정 노이즈 감소
            
        H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
        
        # 칼만 게인 계산
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 혁신 계산 및 제한
        innovation = measurement - H @ self.state
        max_innovation = 0.05  # 최대 혁신 크기 감소
        innovation_magnitude = np.linalg.norm(innovation)
        if innovation_magnitude > max_innovation:
            innovation = innovation * (max_innovation / innovation_magnitude)
        
        # 이전 보정과의 연속성 고려
        correction = K @ innovation
        correction_diff = correction[:2] - self.prev_correction
        if np.linalg.norm(correction_diff) > max_innovation:
            correction_scale = max_innovation / np.linalg.norm(correction_diff)
            correction = np.concatenate([
                self.prev_correction + correction_diff * correction_scale,
                correction[2:]
            ])
        
        self.state = self.state + correction
        self.P = (np.eye(4) - K @ H) @ self.P
        self.prev_correction = correction[:2]

def apply_improved_correction(vehicle_state: VehicleState,
                           kalman_filter: ImprovedKalmanFilter,
                           ramp_curve: 'ParametricRamp',
                           t: float,
                           dt: float) -> VehicleState:
    """개선된 보정 시스템 적용"""
    
    # 1. 현재 GPS 위치를 곡선에 투영
    t_proj, projected_point = ramp_curve.project_point(
        vehicle_state.gps_position, 
        t_guess=t
    )
    
    # 2. 곡선의 기하학적 특성 계산
    curve_point = ramp_curve(t_proj)
    tangent = ramp_curve.derivative(t_proj)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-6:
        tangent = tangent / tangent_norm
    normal = np.array([-tangent[1], tangent[0]])
    
    # 3. GPS 오차 분해
    error_vector = vehicle_state.gps_position - curve_point
    tangential_error = np.dot(error_vector, tangent)
    normal_error = np.dot(error_vector, normal)
    
    # 4. 보정 벡터 계산
    is_ramp = ramp_curve.is_ramp_section(t_proj)
    if is_ramp:
        # 램프 구간에서는 곡선을 따르도록 강한 보정
        correction = -normal_error * normal * 0.8
    else:
        # 직선 구간에서는 약한 보정
        correction = -error_vector * 0.3
    
    # 5. 칼만 필터 업데이트
    measurement = curve_point + correction
    measurement_noise = compute_adaptive_measurement_noise(
        error_vector, tangent, is_ramp
    )
    
    kalman_filter.predict(dt, np.zeros(2))
    kalman_filter.update(measurement, measurement_noise)
    
    # 6. 새로운 상태 생성
    new_state = VehicleState(
        kalman_filter.state[:2],
        kalman_filter.state[2:]
    )
    new_state.gps_position = vehicle_state.gps_position
    
    return new_state

def compute_adaptive_measurement_noise(error_vector: np.ndarray,
                                    path_derivative: np.ndarray,
                                    is_ramp: bool) -> np.ndarray:
    """적응형 측정 노이즈 계산
    Args:
        error_vector: GPS 오차 벡터
        path_derivative: 경로의 접선 벡터
        is_ramp: 램프 구간 여부
    """
    error_magnitude = np.linalg.norm(error_vector)
    path_direction = path_derivative / (np.linalg.norm(path_derivative) + 1e-10)
    
    # 기본 노이즈 설정
    base_noise = 0.1 + 0.1 * error_magnitude
    
    # 방향별 노이즈 계산
    path_normal = np.array([-path_direction[1], path_direction[0]])
    
    # 램프 구간에서는 더 작은 노이즈 (더 강한 보정)
    tangential_noise = base_noise * (0.5 if is_ramp else 1.0)
    normal_noise = base_noise * (0.3 if is_ramp else 1.0)
    
    # 방향별 노이즈 행렬 구성
    R = (np.outer(path_direction, path_direction) * tangential_noise +
         np.outer(path_normal, path_normal) * normal_noise)
    
    return R

def compute_derivative(path_func: Callable, t: float, h: float = 1e-6) -> np.ndarray:
    """경로 함수의 미분 계산"""
    return (path_func(t + h) - path_func(t - h)) / (2 * h)

def compute_second_derivative(path_func: Callable, t: float, h: float = 1e-6) -> np.ndarray:
    """경로 함수의 2차 미분 계산"""
    return (path_func(t + h) - 2 * path_func(t) + path_func(t - h)) / (h * h)

def estimate_acceleration(position: np.ndarray,
                        velocity: np.ndarray,
                        target_position: np.ndarray,
                        path_derivative: np.ndarray) -> np.ndarray:
    """목표 지점을 향한 가속도 추정"""
    # 목표 지점까지의 방향과 거리
    to_target = target_position - position
    distance = np.linalg.norm(to_target)
    
    if distance < 1e-6:
        return np.zeros(2)
    
    # 경로 접선 방향
    path_tangent = path_derivative / np.linalg.norm(path_derivative)
    
    # 목표 방향 가속도
    target_direction = to_target / distance
    acceleration = 2.0 * target_direction - 0.5 * velocity
    
    # 경로 접선 방향 성분 강화
    tangential_component = np.dot(acceleration, path_tangent) * path_tangent
    normal_component = acceleration - tangential_component
    
    return tangential_component + 0.5 * normal_component

def compute_adaptive_measurement_noise(gps_position: np.ndarray,
                                    path_point: np.ndarray,
                                    path_derivative: np.ndarray) -> np.ndarray:
    """GPS 신뢰도 기반 적응형 측정 노이즈"""
    # 입력값이 스칼라인 경우 처리
    if not isinstance(gps_position, np.ndarray):
        gps_position = np.array(gps_position)
    if not isinstance(path_point, np.ndarray):
        path_point = np.array(path_point)
    if not isinstance(path_derivative, np.ndarray):
        path_derivative = np.array(path_derivative)
        
    distance_to_path = np.linalg.norm(gps_position - path_point)
    
    # 거리에 따른 기본 노이즈 설정
    base_noise = 0.1 + 0.2 * np.tanh(distance_to_path)
    
    # 방향 기반 비등방성 노이즈
    path_derivative_norm = np.linalg.norm(path_derivative)
    if path_derivative_norm < 1e-6:
        return np.eye(2) * base_noise
    
    # 2차원 벡터로 변환
    if path_derivative.size == 1:
        path_direction = np.array([1.0, 0.0])
    else:
        path_direction = path_derivative / path_derivative_norm
        if path_direction.size > 2:
            path_direction = path_direction[:2]
    
    path_normal = np.array([-path_direction[1], path_direction[0]])
    
    # 방향별 노이즈 행렬
    R = (np.outer(path_direction, path_direction) * base_noise + 
         np.outer(path_normal, path_normal) * base_noise * 1.5)
    
    return R 