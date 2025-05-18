# 전체 실행 로직
import numpy as np
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 절대 경로로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from spline_curve import fit_spline
from recovery_algo import recovery_step
from data_processing.ramp_data_processor import RampDataProcessor
from visualization.ramp_visualizer import RampVisualizer

def main():
    # 데이터 처리
    data_path = os.path.join(project_root, 'data', 'raw', 'reference_paths', 'reference_data.json')
    processor = RampDataProcessor(data_path)
    ramp_data = processor.process_data()
    
    # 스플라인 피팅
    r_func = fit_spline(ramp_data['x_smooth'], ramp_data['y_smooth'])
    
    # 초기 위치 설정 (첫 번째 데이터 포인트에서 약간 벗어난 위치)
    p = np.array([ramp_data['x_raw'][0] + 0.5, ramp_data['y_raw'][0] + 0.5])
    
    # 보정 과정 기록
    original_positions = [p.copy()]
    corrected_positions = [p.copy()]
    
    # 보정 반복
    for step in range(20):
        p = recovery_step(p, r_func, t=step, alpha=0.7, beta=0.3, eta=1.0, delta_t=0.1)
        original_positions.append(p.copy())
        corrected_positions.append(p.copy())
        print(f"Step {step}: 위치 = {p}")
    
    # 시각화
    visualizer = RampVisualizer()
    
    # 정적 플롯
    visualizer.plot_ramp_data(ramp_data)
    visualizer.save_plot(os.path.join(project_root, 'results', 'ramp_analysis.png'))
    
    # 애니메이션
    anim = visualizer.plot_correction_animation(
        original_positions,
        corrected_positions,
        ramp_data
    )
    
    # 결과 저장 및 표시
    anim.save(os.path.join(project_root, 'results', 'correction_animation.gif'), writer='pillow')
    visualizer.show_plot()

if __name__ == '__main__':
    main()