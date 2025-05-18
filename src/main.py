# 전체 실행 로직
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_processing.ramp_estimator import RampEstimator
from data_processing.gps_3d_error import GPS3DDistortionSimulator
from data_processing.gps_correction import GPSCorrector, GPSMeasurement, VehicleState
from typing import Tuple, Optional
import time

def create_simulation_path(ramp: RampEstimator, t: float, dt: float, 
                         prev_position: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create simulation path point
    Args:
        ramp: RampEstimator instance
        t: Current normalized time (0~1)
        dt: Time step
        prev_position: Previous position (if available)
    Returns:
        Tuple of (position, velocity)
    """
    if t <= 1.0:
        position = ramp.get_position(t)
        velocity = ramp.get_tangent(t) * 10.0  # 10 m/s speed
    else:
        position = prev_position if prev_position is not None else ramp.get_position(1.0)
        velocity = np.zeros(3)
    return position, velocity

def main():
    print("Program starting...")
    start_time = time.time()
    
    # 1. 램프 구조 추정 (실제 데이터 사용)
    print("1. Loading ramp data...")
    reference_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'data', 'raw', 'reference_paths', 'reference_data.json')
    if not os.path.exists(reference_path):
        print(f"Error: File not found: {reference_path}")
        return
        
    try:
        ramp = RampEstimator(reference_path)
        print("Ramp data loaded successfully")
    except Exception as e:
        print(f"Error loading ramp data: {e}")
        return
    
    # 2. GPS 시뮬레이터 및 보정기 초기화
    print("2. Initializing GPS simulator...")
    gps_simulator = GPS3DDistortionSimulator(
        horizontal_std=4.0,  # 수평 오차 증가
        vertical_std=6.0,    # 수직 오차 증가
        correlation_time=2.0  # 시간 상관관계 감소
    )
    
    gps_corrector = GPSCorrector(
        window_size=5,  # 윈도우 크기 증가
        max_acceleration=1.0,  # 가속도 제한 감소
        ramp_constraint_weight=1.5  # 램프 제약 가중치 조정
    )
    print("GPS simulator initialized")

    # 3. 시뮬레이션 파라미터
    print("3. Starting simulation...")
    dt = 0.5  # GPS 측정 간격 증가 (0.5초)
    simulation_time = 20.0
    n_steps = int(simulation_time / dt)
    
    # 4. 데이터 저장용 배열
    true_positions = np.zeros((n_steps, 3))
    raw_gps_positions = np.zeros((n_steps, 3))
    corrected_positions = np.zeros((n_steps, 3))
    timestamps = np.zeros(n_steps)
    
    # 램프까지의 최단 거리를 계산하는 함수
    def ramp_distance(pos, return_closest_point: bool = False):
        _, closest_point = ramp.get_closest_point(pos)
        distance = np.linalg.norm(pos - closest_point)
        if return_closest_point:
            return distance, closest_point
        return distance
    
    # 5. 시뮬레이션 실행
    print("Simulating...")
    for step in range(n_steps):
        if step % 10 == 0:
            print(f"Progress: {step}/{n_steps} ({step/n_steps*100:.1f}%)")
            
        # 현재 시간
        t = step * dt / simulation_time
        current_time = step * dt
        
        # 현재 위치와 속도 계산
        position, velocity = create_simulation_path(
            ramp, t, dt,
            prev_position=true_positions[step-1] if step > 0 else None
        )
            
        # GPS 측정값 생성 및 보정
        raw_gps = gps_simulator.apply_distortion(position, position[2])
        
        measurement = GPSMeasurement(
            position=raw_gps,
            timestamp=current_time,
            accuracy=2.0
        )
        gps_corrector.add_measurement(measurement)
        
        corrected_position = gps_corrector.correct_position(ramp_distance)
        
        # 데이터 저장
        true_positions[step] = position
        raw_gps_positions[step] = raw_gps
        corrected_positions[step] = corrected_position
        timestamps[step] = current_time
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds")
    
    # 6. 결과 시각화
    print("\n6. Visualizing results...")
    
    # 3D 그래프
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 개선된 시각화 사용
    ramp.plot_results(
        raw_gps=raw_gps_positions[::2],  # 데이터 간격 증가로 가시성 향상
        corrected_gps=corrected_positions[::2]
    )
    
    # 결과 저장
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'ramp_3d_simulation.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 오차 분석
    raw_errors = np.linalg.norm(raw_gps_positions - true_positions, axis=1)
    corrected_errors = np.linalg.norm(corrected_positions - true_positions, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, raw_errors, 'r:', label='Raw GPS Error', alpha=0.5, linewidth=2)
    plt.plot(timestamps, corrected_errors, 'g-', label='Corrected Error', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Error (m)', fontsize=12)
    plt.title('Position Error Over Time', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 오차 통계
    print("\nError Statistics:")
    print(f"Raw GPS - Mean: {np.mean(raw_errors):.2f}m, Max: {np.max(raw_errors):.2f}m")
    print(f"Corrected - Mean: {np.mean(corrected_errors):.2f}m, Max: {np.max(corrected_errors):.2f}m")
    print(f"Improvement: {(1 - np.mean(corrected_errors)/np.mean(raw_errors))*100:.1f}%")
    
    plt.savefig(os.path.join('results', 'error_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Program completed successfully")

if __name__ == '__main__':
    main()