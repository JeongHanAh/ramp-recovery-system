import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing.road_geometry import RampExtractor
from curve_modeling.bspline import RampCurve
from path_recovery.recovery import PathRecovery
from data_processing.sensor_data import SensorDataProcessor

def plot_road_and_ramp(df, ramp_segments, ramp_curves):
    """도로와 추출된 램프 구간 시각화"""
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
                      f'길이: {segment.length:.1f}m\n'
                      f'평균 곡률: {segment.mean_curvature:.3f}\n'
                      f'평균 경사도: {segment.mean_slope:.3f}')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('도로 데이터와 추출된 램프 구간')
    
    # 곡률과 경사도 프로파일 플롯
    plt.subplot(212)
    x_dist = np.cumsum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2))
    x_dist = np.concatenate([[0], x_dist])
    
    # 곡률 계산
    curvature = extractor.calculate_curvature(df['x'].values, df['y'].values)
    slope = extractor.calculate_slope(df['x'].values, df['y'].values, df['z'].values)
    
    plt.plot(x_dist, curvature, 'b-', label='곡률', alpha=0.5)
    plt.plot(x_dist, slope, 'r-', label='경사도', alpha=0.5)
    
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
    plt.show()

def simulate_recovery(ramp_curve, initial_offset=5.0):
    """경로 복구 시뮬레이션"""
    # 경로 복구 객체 생성
    recovery = PathRecovery(ramp_curve)
    
    # 시뮬레이션 초기 조건
    t = np.linspace(0, 1, 100)
    ideal_path = ramp_curve.evaluate(t)
    
    # 초기 위치를 이상적 경로에서 offset만큼 이동
    current_position = ideal_path[0] + np.array([0, initial_offset])
    vehicle_heading = 0.0
    
    # 시뮬레이션 결과 저장
    positions = [current_position]
    headings = [vehicle_heading]
    steering_angles = []
    
    # 경로 복구 시뮬레이션
    dt = 0.1
    velocity = 10.0  # m/s
    for _ in range(50):
        # 조향각 계산
        steering_angle = recovery.get_steering_angle(
            current_position, vehicle_heading
        )
        
        # 간단한 차량 운동 모델
        vehicle_heading += steering_angle * dt
        dx = velocity * np.cos(vehicle_heading) * dt
        dy = velocity * np.sin(vehicle_heading) * dt
        current_position += np.array([dx, dy])
        
        positions.append(current_position)
        headings.append(vehicle_heading)
        steering_angles.append(steering_angle)
    
    # 결과 시각화
    positions = np.array(positions)
    
    plt.figure(figsize=(15, 10))
    
    # 경로 플롯
    plt.subplot(211)
    plt.plot(ideal_path[:, 0], ideal_path[:, 1],
            'k--', label='이상적 경로')
    plt.plot(positions[:, 0], positions[:, 1],
            'r.-', label='복구 경로')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('경로 복구 시뮬레이션')
    
    # 조향각 플롯
    plt.subplot(212)
    t = np.arange(len(steering_angles)) * dt
    plt.plot(t, np.degrees(steering_angles), 'b-', label='조향각')
    plt.grid(True)
    plt.legend()
    plt.xlabel('시간 (초)')
    plt.ylabel('조향각 (도)')
    plt.title('조향각 변화')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. 도로 데이터 로드 및 램프 추출
    global extractor  # plot_road_and_ramp 함수에서 사용하기 위해
    extractor = RampExtractor(
        curvature_threshold=0.015,
        slope_threshold=0.03,
        window_size=20,
        min_segment_length=50.0
    )
    
    try:
        df, optional_data = extractor.load_road_data('data/road_geometry.csv')
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        print("테스트용 더미 데이터를 생성합니다...")
        
        # 테스트용 더미 데이터 생성
        x = np.linspace(0, 100, 1000)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        
        # 두 개의 램프 형태 생성
        ramp1_idx = (x > 30) & (x < 70)
        y[ramp1_idx] = 20 * np.sin(np.pi * (x[ramp1_idx] - 30) / 40)
        z[ramp1_idx] = 5 * np.sin(np.pi * (x[ramp1_idx] - 30) / 40)
        
        ramp2_idx = (x > 150) & (x < 200)
        y[ramp2_idx] = -15 * np.sin(np.pi * (x[ramp2_idx] - 150) / 50)
        z[ramp2_idx] = 3 * np.sin(np.pi * (x[ramp2_idx] - 150) / 50)
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        optional_data = None
    
    # 2. 램프 구간 추출
    ramp_segments = extractor.extract_ramp_segments(df, optional_data)
    
    if not ramp_segments:
        print("램프 구간을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(ramp_segments)}개의 램프 구간을 찾았습니다:")
    for i, segment in enumerate(ramp_segments):
        print(f"\n램프 {i+1}:")
        print(f"  길이: {segment.length:.1f}m")
        print(f"  평균 곡률: {segment.mean_curvature:.3f}")
        print(f"  최대 곡률: {segment.max_curvature:.3f}")
        print(f"  평균 경사도: {segment.mean_slope:.3f}")
        print(f"  최대 경사도: {segment.max_slope:.3f}")
    
    # 3. 각 램프 구간에 대해 B-스플라인 피팅
    ramp_curves = []
    for segment in ramp_segments:
        control_points = extractor.get_ramp_control_points(df, segment)
        ramp_curves.append(RampCurve(control_points))
    
    # 4. 결과 시각화
    plot_road_and_ramp(df, ramp_segments, ramp_curves)
    
    # 5. 첫 번째 램프에 대해 경로 복구 시뮬레이션
    if ramp_curves:
        simulate_recovery(ramp_curves[0])

if __name__ == '__main__':
    main() 