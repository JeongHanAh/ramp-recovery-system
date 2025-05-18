import numpy as np
import json
from typing import Tuple, List, Dict
from scipy.signal import savgol_filter

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