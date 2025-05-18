import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Callable
import json
import os

class ParametricRamp:
    def __init__(self, ramp_id: str = "ramp1"):
        """파라메트릭 램프 곡선 초기화
        Args:
            ramp_id: 사용할 램프 ID
        """
        # 실제 램프 데이터 로드
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'data', 'road_geometry.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # 지정된 램프 데이터 찾기
        ramp_data = next(seg for seg in data['segments'] if seg['id'] == ramp_id)
        coords = np.array(ramp_data['coordinates'])
        
        # 전체 거리 계산
        distances = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # 시작점 추가
        total_distance = distances[-1]
        
        # 정규화된 매개변수 생성
        t_points = distances / total_distance
        
        # 보간 함수 생성
        self.interp_x = interp1d(t_points, coords[:, 0], kind='cubic')
        self.interp_y = interp1d(t_points, coords[:, 1], kind='cubic')
        
        # 램프 구간 식별
        self.ramp_start = 0.2  # 20% 지점부터 램프 시작
        self.ramp_end = 0.8    # 80% 지점에서 램프 종료
        
    def __call__(self, t: float) -> np.ndarray:
        """주어진 매개변수 t에서의 곡선 위치 반환"""
        t = np.clip(t, 0, 1)
        return np.array([float(self.interp_x(t)), float(self.interp_y(t))])
    
    def derivative(self, t: float, h: float = 1e-6) -> np.ndarray:
        """곡선의 도함수 (접선 벡터) 계산"""
        t = np.clip(t, 0, 1)
        t_plus = min(t + h, 1)
        t_minus = max(t - h, 0)
        return (self(t_plus) - self(t_minus)) / (t_plus - t_minus)
    
    def second_derivative(self, t: float, h: float = 1e-6) -> np.ndarray:
        """곡선의 2차 도함수 계산"""
        t = np.clip(t, 0, 1)
        t_plus = min(t + h, 1)
        t_minus = max(t - h, 0)
        der_plus = self.derivative(t_plus, h)
        der_minus = self.derivative(t_minus, h)
        return (der_plus - der_minus) / (t_plus - t_minus)
    
    def curvature(self, t: float) -> float:
        """곡선의 곡률 계산"""
        der1 = self.derivative(t)
        der2 = self.second_derivative(t)
        
        numerator = der1[0] * der2[1] - der1[1] * der2[0]
        denominator = np.power(np.sum(der1**2), 1.5)
        
        return numerator / denominator if denominator > 1e-10 else 0.0
    
    def is_ramp_section(self, t: float) -> bool:
        """현재 위치가 램프 구간인지 확인"""
        return self.ramp_start <= t <= self.ramp_end
    
    def project_point(self, point: np.ndarray, 
                     t_guess: float = None, 
                     num_iterations: int = 10) -> Tuple[float, np.ndarray]:
        """주어진 점을 곡선에 투영"""
        if t_guess is None:
            # 초기 추정값 계산
            t_samples = np.linspace(0, 1, 100)
            distances = [np.linalg.norm(self(t) - point) for t in t_samples]
            t = t_samples[np.argmin(distances)]
        else:
            t = t_guess
        
        # Newton-Raphson 방법으로 최적의 t 찾기
        for _ in range(num_iterations):
            pos = self(t)
            der = self.derivative(t)
            der2 = self.second_derivative(t)
            
            f = np.dot(pos - point, der)
            f_prime = np.dot(der, der) + np.dot(pos - point, der2)
            
            if abs(f_prime) < 1e-10:
                break
                
            t_new = t - f / f_prime
            if abs(t_new - t) < 1e-10:
                break
                
            t = np.clip(t_new, 0, 1)
        
        return t, self(t)
    
    def get_path_coordinates(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """시각화를 위한 곡선 좌표 생성"""
        t = np.linspace(0, 1, num_points)
        points = np.array([self(ti) for ti in t])
        return points[:, 0], points[:, 1] 