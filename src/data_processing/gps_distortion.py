import numpy as np
from typing import Tuple, List, Callable

class GPSDistortionSimulator:
    def __init__(self, noise_level: float = 0.02, bias_factor: float = 0.03):
        """GPS 왜곡 시뮬레이터 초기화
        Args:
            noise_level: 랜덤 노이즈 크기
            bias_factor: 시스템적 바이어스 크기
        """
        self.noise_level = noise_level
        self.bias_factor = bias_factor
        self.bias = np.random.normal(0, bias_factor, size=2)  # 고정된 바이어스
        
    def apply_distortion(self, true_position: np.ndarray) -> np.ndarray:
        """GPS 왜곡 적용
        Args:
            true_position: 실제 위치 [x, y]
        Returns:
            왜곡된 GPS 위치
        """
        # 랜덤 노이즈 생성
        noise = np.random.normal(0, self.noise_level, size=2)
        
        # 위치 기반 시스템적 바이어스
        position_bias = self.bias * np.linalg.norm(true_position) * 0.01
        
        # 시간에 따라 변하는 동적 바이어스
        dynamic_bias = np.array([
            np.sin(np.linalg.norm(true_position) * 0.1),
            np.cos(np.linalg.norm(true_position) * 0.1)
        ]) * self.bias_factor
        
        # 왜곡 적용
        distorted_position = true_position + noise + position_bias + dynamic_bias
        
        return distorted_position
    
    def generate_distorted_path(self, 
                              reference_func: Callable,
                              t_range: Tuple[float, float],
                              n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전체 경로에 대한 GPS 왜곡 생성
        
        Args:
            reference_func: 기준 경로 함수
            t_range: 시간 범위 (시작, 끝)
            n_points: 생성할 포인트 수
            
        Returns:
            (x_true, y_true, x_distorted, y_distorted)
        """
        t = np.linspace(t_range[0], t_range[1], n_points)
        x_true = []
        y_true = []
        x_distorted = []
        y_distorted = []
        
        for ti in t:
            point = reference_func(ti)
            x_true.append(point[0])
            y_true.append(point[1])
            
            distorted_point = self.apply_distortion(point)
            x_distorted.append(distorted_point[0])
            y_distorted.append(distorted_point[1])
        
        return (np.array(x_true), np.array(y_true),
                np.array(x_distorted), np.array(y_distorted)) 