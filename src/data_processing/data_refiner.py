import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from pyproj import Transformer
import logging
from pathlib import Path

class RoadGeometryRefiner:
    """도로 기하 데이터 정제를 위한 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        매개변수:
            config (Dict): 정제 설정
                - outlier_std_threshold (float): 이상치 탐지를 위한 표준편차 임계값
                - smoothing_window (int): 스무딩 윈도우 크기
                - min_point_distance (float): 최소 점 간 거리 (미터)
                - max_point_distance (float): 최대 점 간 거리 (미터)
        """
        self.config = {
            'outlier_std_threshold': 3.0,
            'smoothing_window': 21,
            'smoothing_order': 3,
            'min_point_distance': 0.1,  # 미터
            'max_point_distance': 5.0,  # 미터
        }
        if config:
            self.config.update(config)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 탐지 및 제거
        
        매개변수:
            df (pd.DataFrame): 입력 데이터프레임
            
        반환:
            pd.DataFrame: 이상치가 제거된 데이터프레임
        """
        # 3D 좌표에 대한 이상치 탐지
        coords = df[['x', 'y', 'z']].values
        
        # 각 점에서 가장 가까운 이웃까지의 거리 계산
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # 거리 기반 이상치 탐지
        distance_mask = (
            (min_distances > self.config['min_point_distance']) &
            (min_distances < self.config['max_point_distance'])
        )
        
        # 통계 기반 이상치 탐지
        stat_masks = {}
        for col in ['x', 'y', 'z', 'curvature', 'slope']:
            if col in df.columns:
                values = df[col].values
                mean = np.mean(values)
                std = np.std(values)
                stat_masks[col] = (
                    np.abs(values - mean) <= 
                    self.config['outlier_std_threshold'] * std
                )
        
        # 모든 마스크 결합
        final_mask = distance_mask
        for mask in stat_masks.values():
            final_mask = final_mask & mask
        
        removed_count = len(df) - final_mask.sum()
        self.logger.info(f"이상치 {removed_count}개 제거됨")
        
        return df[final_mask].copy()
    
    def smooth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 스무딩
        
        매개변수:
            df (pd.DataFrame): 입력 데이터프레임
            
        반환:
            pd.DataFrame: 스무딩된 데이터프레임
        """
        smoothed_df = df.copy()
        
        # 좌표 데이터 스무딩
        for col in ['x', 'y', 'z']:
            smoothed_df[col] = savgol_filter(
                df[col],
                window_length=self.config['smoothing_window'],
                polyorder=self.config['smoothing_order']
            )
        
        # 곡률과 경사도가 있다면 스무딩
        for col in ['curvature', 'slope']:
            if col in df.columns:
                smoothed_df[col] = savgol_filter(
                    df[col],
                    window_length=self.config['smoothing_window'],
                    polyorder=self.config['smoothing_order']
                )
        
        self.logger.info("데이터 스무딩 완료")
        return smoothed_df
    
    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        누락된 데이터 보간
        
        매개변수:
            df (pd.DataFrame): 입력 데이터프레임
            
        반환:
            pd.DataFrame: 보간된 데이터프레임
        """
        # 누락값이 있는 컬럼 찾기
        cols_with_missing = df.columns[df.isna().any()].tolist()
        
        if cols_with_missing:
            self.logger.info(f"누락값이 있는 컬럼: {cols_with_missing}")
            
            # 3차 스플라인 보간
            df_interpolated = df.copy()
            for col in cols_with_missing:
                df_interpolated[col] = df[col].interpolate(
                    method='cubic',
                    limit_direction='both'
                )
            
            missing_counts = df[cols_with_missing].isna().sum()
            self.logger.info(f"보간된 누락값 수: {dict(missing_counts)}")
            
            return df_interpolated
        
        return df
    
    def transform_coordinates(self, df: pd.DataFrame,
                            from_crs: str,
                            to_crs: str) -> pd.DataFrame:
        """
        좌표계 변환
        
        매개변수:
            df (pd.DataFrame): 입력 데이터프레임
            from_crs (str): 원본 좌표계 (예: "EPSG:4326")
            to_crs (str): 대상 좌표계 (예: "EPSG:32652")
            
        반환:
            pd.DataFrame: 변환된 데이터프레임
        """
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        df_transformed = df.copy()
        
        # x, y 좌표 변환
        x_transformed, y_transformed = transformer.transform(
            df['x'].values,
            df['y'].values
        )
        
        df_transformed['x'] = x_transformed
        df_transformed['y'] = y_transformed
        
        self.logger.info(f"좌표계 변환 완료: {from_crs} → {to_crs}")
        return df_transformed
    
    def refine_data(self, input_path: str,
                    output_path: str,
                    from_crs: Optional[str] = None,
                    to_crs: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        데이터 정제 파이프라인 실행
        
        매개변수:
            input_path (str): 입력 CSV 파일 경로
            output_path (str): 출력 CSV 파일 경로
            from_crs (str, optional): 원본 좌표계
            to_crs (str, optional): 대상 좌표계
            
        반환:
            Tuple[pd.DataFrame, Dict]: (정제된 데이터프레임, 처리 통계)
        """
        # 입력 파일 확인
        if not Path(input_path).exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
        
        # 데이터 로드
        self.logger.info(f"데이터 로드 중: {input_path}")
        df = pd.read_csv(input_path)
        initial_count = len(df)
        
        # 정제 파이프라인 실행
        stats = {'initial_count': initial_count}
        
        # 1. 누락값 보간
        df = self.interpolate_missing(df)
        stats['interpolated_count'] = len(df)
        
        # 2. 이상치 제거
        df = self.detect_outliers(df)
        stats['after_outlier_removal'] = len(df)
        
        # 3. 좌표계 변환 (필요한 경우)
        if from_crs and to_crs:
            df = self.transform_coordinates(df, from_crs, to_crs)
        
        # 4. 데이터 스무딩
        df = self.smooth_data(df)
        
        # 결과 저장
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"정제된 데이터 저장됨: {output_path}")
        
        # 처리 통계 업데이트
        stats['final_count'] = len(df)
        stats['removed_points'] = initial_count - len(df)
        stats['removal_percentage'] = (
            (initial_count - len(df)) / initial_count * 100
        )
        
        return df, stats 