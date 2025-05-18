import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import json
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev
from datetime import datetime

class RampExtractor:
    def __init__(self, data_path: str):
        """램프 구조 추출기 초기화
        Args:
            data_path: 도로 기하 데이터 파일 경로 (CSV 또는 JSON)
        """
        self.data_path = Path(data_path)
        self.ramp_structures = []
        self.current_ramp = None
    
    def load_road_data(self) -> None:
        """도로 기하 데이터 로드"""
        if self.data_path.suffix.lower() == '.csv':
            self._load_csv_data()
        else:
            self._load_json_data()
    
    def _convert_time_to_seconds(self, time_str: str) -> float:
        """시간 문자열을 초 단위로 변환"""
        try:
            # 시간 문자열에서 공백 제거
            time_str = time_str.strip()
            # 시간 파싱
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
            # 초 단위로 변환
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond/1000000
        except ValueError:
            return 0.0
    
    def _load_csv_data(self) -> None:
        """CSV 형식의 도로 기하 데이터 로드"""
        # CSV 파일 읽기
        df = pd.read_csv(self.data_path)
        print("CSV 파일 로드 완료")
        print(f"데이터 컬럼: {df.columns.tolist()}")
        print(f"데이터 크기: {len(df)} 행")
        
        # 시간 데이터를 초 단위로 변환
        df['TimeSeconds'] = df['Time'].apply(self._convert_time_to_seconds)
        print("시간 변환 완료")
        
        # 데이터 전처리 및 구조화
        segments = []
        
        # 시간을 기준으로 데이터 분할 (예: 10초 간격)
        time_window = 10  # 10초 단위로 분할
        df['TimeGroup'] = (df['TimeSeconds'] // time_window).astype(int)
        
        # 시간 그룹별로 처리
        for group_id, group in df.groupby('TimeGroup'):
            # 좌표 추출 (Lon, Lat 사용)
            coordinates = []
            for _, row in group.iterrows():
                try:
                    # 실제 위치 데이터 사용
                    x = float(row['Lon'])  # 경도
                    y = float(row['Altitude'])  # 고도
                    coordinates.append([x, y])
                except (ValueError, KeyError):
                    continue
            
            if len(coordinates) > 2:  # 최소 3개 이상의 포인트가 있는 경우만 포함
                segments.append({
                    'id': f'segment_{group_id}',
                    'coordinates': coordinates
                })
        
        print(f"생성된 세그먼트 수: {len(segments)}")
        self.raw_data = {'segments': segments}
    
    def _load_json_data(self) -> None:
        """JSON 형식의 도로 기하 데이터 로드"""
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
    
    def identify_ramps(self, 
                      min_curvature: float = 0.1,
                      min_length: float = 50.0,
                      max_length: float = 500.0) -> List[Dict]:
        """램프 구조 식별
        Args:
            min_curvature: 최소 곡률 임계값
            min_length: 최소 램프 길이 (미터)
            max_length: 최대 램프 길이 (미터)
        Returns:
            식별된 램프 구조 목록
        """
        ramps = []
        
        for segment in self.raw_data['segments']:
            coords = np.array(segment['coordinates'])
            
            # 곡률 계산
            dx = np.gradient(coords[:, 0])
            dy = np.gradient(coords[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
            
            # 길이 계산
            length = sum(euclidean(coords[i], coords[i+1]) 
                       for i in range(len(coords)-1))
            
            # 램프 조건 확인
            if (np.mean(curvature) > min_curvature and
                min_length <= length <= max_length):
                
                # 정규화된 좌표로 변환
                normalized_coords = self._normalize_coordinates(coords)
                
                ramps.append({
                    'coordinates': normalized_coords.tolist(),
                    'length': length,
                    'mean_curvature': float(np.mean(curvature))
                })
        
        self.ramp_structures = ramps
        return ramps
    
    def select_random_ramp(self) -> np.ndarray:
        """무작위 램프 구조 선택"""
        if not self.ramp_structures:
            raise ValueError("램프 구조를 먼저 식별해주세요.")
        
        self.current_ramp = np.array(
            np.random.choice(self.ramp_structures)['coordinates']
        )
        return self.current_ramp
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """좌표 정규화 및 시작점 조정
        Args:
            coords: 원본 좌표 배열
        Returns:
            정규화된 좌표 배열
        """
        # 시작점을 원점으로 이동
        normalized = coords - coords[0]
        
        # 전체 크기를 1로 스케일링
        max_dim = max(normalized.max(axis=0) - normalized.min(axis=0))
        normalized = normalized / max_dim
        
        # 우회전 램프의 경우 좌표계 조정
        if normalized[-1, 0] < 0:
            normalized[:, 0] = -normalized[:, 0]
        
        return normalized
    
    def get_ramp_function(self) -> callable:
        """현재 선택된 램프의 함수 표현 반환"""
        if self.current_ramp is None:
            raise ValueError("램프를 먼저 선택해주세요.")
        
        # B-스플라인 피팅
        tck, _ = splprep([self.current_ramp[:, 0], 
                         self.current_ramp[:, 1]], s=0)
        
        def ramp_function(t: float) -> np.ndarray:
            """램프 곡선 상의 점 계산
            Args:
                t: 곡선 파라미터 (0~1)
            Returns:
                [x, y] 좌표
            """
            x, y = splev(t, tck)
            return np.array([float(x), float(y)])
        
        return ramp_function 