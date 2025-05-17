import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from data_processing.data_refiner import RoadGeometryRefiner
from data_processing.road_geometry import RampExtractor
from curve_modeling.bspline import RampCurve

class DataPipeline:
    """도로 기하 데이터 처리 파이프라인"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        매개변수:
            config_path (str, optional): 설정 파일 경로
        """
        # 기본 설정
        self.config = {
            'data_dir': 'data',
            'raw_data_dir': 'data/raw',
            'refined_data_dir': 'data/refined',
            'results_dir': 'data/results',
            'from_crs': 'EPSG:4326',  # WGS84
            'to_crs': 'EPSG:32652',   # UTM Zone 52N
            'refiner_config': {
                'outlier_std_threshold': 3.0,
                'smoothing_window': 21,
                'min_point_distance': 0.1,
                'max_point_distance': 5.0
            },
            'ramp_extractor_config': {
                'curvature_threshold': 0.015,
                'slope_threshold': 0.03,
                'window_size': 20,
                'min_segment_length': 50.0
            }
        }
        
        # 설정 파일이 있으면 로드
        if config_path:
            self._load_config(config_path)
        
        # 디렉토리 생성
        for dir_key in ['data_dir', 'raw_data_dir', 'refined_data_dir', 'results_dir']:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _load_config(self, config_path: str):
        """설정 파일 로드"""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            print("기본 설정을 사용합니다.")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('DataPipeline')
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = Path(self.config['results_dir']) / 'pipeline.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def process_file(self, input_file: str) -> Dict:
        """
        단일 파일 처리
        
        매개변수:
            input_file (str): 입력 파일 경로
            
        반환:
            Dict: 처리 결과 통계
        """
        self.logger.info(f"파일 처리 시작: {input_file}")
        
        # 출력 파일 경로 설정
        input_path = Path(input_file)
        refined_path = Path(self.config['refined_data_dir']) / f"refined_{input_path.name}"
        results_path = Path(self.config['results_dir']) / input_path.stem
        results_path.mkdir(exist_ok=True)
        
        try:
            # 1. 데이터 정제
            refiner = RoadGeometryRefiner(self.config['refiner_config'])
            df, refine_stats = refiner.refine_data(
                str(input_path),
                str(refined_path),
                self.config['from_crs'],
                self.config['to_crs']
            )
            
            # 2. 램프 구간 추출
            extractor = RampExtractor(**self.config['ramp_extractor_config'])
            ramp_segments = extractor.extract_ramp_segments(df)
            
            # 3. B-스플라인 피팅 및 시각화
            ramp_curves = []
            for segment in ramp_segments:
                control_points = extractor.get_ramp_control_points(df, segment)
                ramp_curves.append(RampCurve(control_points))
            
            # 결과 시각화
            self._visualize_results(
                df, ramp_segments, ramp_curves,
                str(results_path / 'visualization.png')
            )
            
            # 처리 통계 저장
            stats = {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(input_path),
                'refined_file': str(refined_path),
                'refine_stats': refine_stats,
                'ramp_segments': len(ramp_segments),
                'ramp_details': [
                    {
                        'length': segment.length,
                        'mean_curvature': segment.mean_curvature,
                        'max_curvature': segment.max_curvature,
                        'mean_slope': segment.mean_slope,
                        'max_slope': segment.max_slope
                    }
                    for segment in ramp_segments
                ]
            }
            
            # 통계 저장
            with open(results_path / 'stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"파일 처리 완료: {input_file}")
            return stats
            
        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 발생: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _visualize_results(self, df: pd.DataFrame,
                          ramp_segments: list,
                          ramp_curves: list,
                          output_path: str):
        """결과 시각화 및 저장"""
        plt.figure(figsize=(15, 10))
        
        # 전체 도로 데이터 플롯
        plt.subplot(211)
        plt.plot(df['x'], df['y'], 'k.', alpha=0.3, label='도로 데이터')
        
        # 각 램프 구간 플롯
        colors = plt.cm.rainbow(np.linspace(0, 1, len(ramp_segments)))
        for segment, curve, color in zip(ramp_segments, ramp_curves, colors):
            # 실제 램프 데이터
            plt.plot(df['x'][segment.start_idx:segment.end_idx],
                    df['y'][segment.start_idx:segment.end_idx],
                    '.', color=color, alpha=0.5)
            
            # B-스플라인 피팅 결과
            t = np.linspace(0, 1, 100)
            curve_points = curve.evaluate(t)
            plt.plot(curve_points[:, 0], curve_points[:, 1],
                    '-', color=color, linewidth=2,
                    label=f'램프 {len(plt.gca().get_lines())//2}\n'
                          f'길이: {segment.length:.1f}m')
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('도로 데이터와 추출된 램프 구간')
        
        # 프로파일 플롯
        plt.subplot(212)
        x_dist = np.cumsum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2))
        x_dist = np.concatenate([[0], x_dist])
        
        if 'curvature' in df.columns:
            plt.plot(x_dist, df['curvature'], 'b-', label='곡률', alpha=0.5)
        if 'slope' in df.columns:
            plt.plot(x_dist, df['slope'], 'r-', label='경사도', alpha=0.5)
        
        # 램프 구간 표시
        for segment in ramp_segments:
            start_dist = x_dist[segment.start_idx]
            end_dist = x_dist[segment.end_idx]
            plt.axvspan(start_dist, end_dist, color='yellow', alpha=0.2)
        
        plt.grid(True)
        plt.legend()
        plt.xlabel('거리 (m)')
        plt.ylabel('곡률/경사도')
        plt.title('도로 프로파일')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='도로 기하 데이터 처리 파이프라인')
    parser.add_argument('input', help='입력 CSV 파일 또는 디렉토리')
    parser.add_argument('--config', help='설정 파일 경로')
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = DataPipeline(args.config)
    
    # 입력 처리
    input_path = Path(args.input)
    if input_path.is_file():
        # 단일 파일 처리
        pipeline.process_file(str(input_path))
    elif input_path.is_dir():
        # 디렉토리 내 모든 CSV 파일 처리
        for file_path in input_path.glob('*.csv'):
            pipeline.process_file(str(file_path))
    else:
        print(f"잘못된 입력 경로: {args.input}")
        sys.exit(1)

if __name__ == '__main__':
    main() 