import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class RampSegment:
    """램프 구간 정보를 저장하는 데이터 클래스"""
    start_idx: int
    end_idx: int
    mean_curvature: float
    max_curvature: float
    mean_slope: float
    max_slope: float
    length: float

class RampExtractor:
    """도로 기하 데이터에서 램프 구간을 추출하는 클래스"""
    
    def __init__(self, 
                 curvature_threshold: float = 0.01,
                 slope_threshold: float = 0.03,
                 window_size: int = 20,
                 min_segment_length: float = 50.0):  # 미터 단위
        """
        매개변수:
            curvature_threshold (float): 램프 구간으로 판단할 곡률 임계값
            slope_threshold (float): 램프 구간으로 판단할 경사도 임계값
            window_size (int): 곡률과 경사도 계산을 위한 윈도우 크기
            min_segment_length (float): 최소 램프 구간 길이 (미터)
        """
        self.curvature_threshold = curvature_threshold
        self.slope_threshold = slope_threshold
        self.window_size = window_size
        self.min_segment_length = min_segment_length
    
    def load_road_data(self, csv_path: str) -> pd.DataFrame:
        """
        도로 기하 데이터 CSV 파일 로드
        
        매개변수:
            csv_path (str): CSV 파일 경로
            
        반환:
            pd.DataFrame: 도로 기하 데이터
        """
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['x', 'y', 'z']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV 파일에 필요한 컬럼이 없습니다: {required_columns}")
            
            # 추가 컬럼이 있다면 활용
            optional_columns = {
                'curvature': None,
                'radius': None,
                'slope': None
            }
            
            for col in optional_columns:
                if col in df.columns:
                    optional_columns[col] = df[col].values
            
            return df, optional_columns
        except Exception as e:
            raise Exception(f"데이터 로드 중 오류 발생: {str(e)}")
    
    def calculate_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        주어진 점들의 곡률 계산
        
        매개변수:
            x (np.ndarray): x 좌표 배열
            y (np.ndarray): y 좌표 배열
            
        반환:
            np.ndarray: 각 점에서의 곡률
        """
        # 매개변수화된 스플라인 피팅
        t = np.arange(len(x))
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        
        # 1차, 2차 도함수 계산
        dx = cs_x.derivative(1)(t)
        dy = cs_y.derivative(1)(t)
        ddx = cs_x.derivative(2)(t)
        ddy = cs_y.derivative(2)(t)
        
        # 곡률 계산: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
        
        # NaN 값을 0으로 대체
        curvature = np.nan_to_num(curvature, 0)
        
        return curvature
    
    def calculate_slope(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        3D 공간에서의 경사도 계산
        
        매개변수:
            x, y, z (np.ndarray): 3D 좌표
            
        반환:
            np.ndarray: 각 점에서의 경사도
        """
        # 점들 사이의 거리 계산
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        
        # 수평 거리
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        
        # 경사도 계산 (수직 변화 / 수평 거리)
        slope = np.abs(dz) / (horizontal_dist + 1e-10)  # 0으로 나누기 방지
        
        # 처음과 끝 점의 경사도를 이웃한 값으로 설정
        slope = np.concatenate([[slope[0]], slope])
        
        return slope
    
    def calculate_segment_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        구간의 길이 계산
        
        매개변수:
            x, y (np.ndarray): 좌표
            
        반환:
            float: 구간 길이 (미터)
        """
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))
    
    def extract_ramp_segments(self, df: pd.DataFrame, 
                            optional_data: Dict[str, np.ndarray] = None) -> List[RampSegment]:
        """
        도로 데이터에서 램프 구간 추출 (곡률과 경사도 모두 고려)
        
        매개변수:
            df (pd.DataFrame): 도로 기하 데이터
            optional_data (Dict): 추가 데이터 (곡률, 반경, 경사도 등)
            
        반환:
            List[RampSegment]: 램프 구간 정보 리스트
        """
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        
        # 곡률 계산 또는 로드
        if optional_data and optional_data['curvature'] is not None:
            curvature = optional_data['curvature']
        else:
            curvature = self.calculate_curvature(x, y)
        
        # 경사도 계산 또는 로드
        if optional_data and optional_data['slope'] is not None:
            slope = optional_data['slope']
        else:
            slope = self.calculate_slope(x, y, z)
        
        # 이동 평균을 사용한 스무딩
        smoothed_curvature = np.convolve(
            curvature, 
            np.ones(self.window_size)/self.window_size, 
            mode='valid'
        )
        smoothed_slope = np.convolve(
            slope,
            np.ones(self.window_size)/self.window_size,
            mode='valid'
        )
        
        # 램프 구간 식별 (곡률 또는 경사도 조건 만족)
        is_ramp = ((smoothed_curvature > self.curvature_threshold) | 
                  (smoothed_slope > self.slope_threshold))
        
        # 연속된 램프 구간 찾기
        ramp_segments = []
        start_idx = None
        
        for i in range(len(is_ramp)):
            if is_ramp[i] and start_idx is None:
                start_idx = i
            elif (not is_ramp[i] or i == len(is_ramp)-1) and start_idx is not None:
                end_idx = i
                
                # 구간 길이 확인
                segment_length = self.calculate_segment_length(
                    x[start_idx:end_idx],
                    y[start_idx:end_idx]
                )
                
                if segment_length >= self.min_segment_length:
                    segment = RampSegment(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        mean_curvature=float(np.mean(curvature[start_idx:end_idx])),
                        max_curvature=float(np.max(curvature[start_idx:end_idx])),
                        mean_slope=float(np.mean(slope[start_idx:end_idx])),
                        max_slope=float(np.max(slope[start_idx:end_idx])),
                        length=segment_length
                    )
                    ramp_segments.append(segment)
                start_idx = None
        
        return ramp_segments
    
    def get_ramp_control_points(self, df: pd.DataFrame, 
                              segment: RampSegment,
                              n_points: int = 10) -> np.ndarray:
        """
        램프 구간에서 B-스플라인 제어점 추출
        
        매개변수:
            df (pd.DataFrame): 도로 기하 데이터
            segment (RampSegment): 램프 구간 정보
            n_points (int): 추출할 제어점 수
            
        반환:
            np.ndarray: 제어점 배열 (n_points x 2)
        """
        x = df['x'].values[segment.start_idx:segment.end_idx]
        y = df['y'].values[segment.start_idx:segment.end_idx]
        
        # 곡률이 높은 지점에서 더 많은 제어점 선택
        t = np.linspace(0, len(x)-1, n_points)
        indices = np.round(t).astype(int)
        control_points = np.column_stack([x[indices], y[indices]])
        
        return control_points 