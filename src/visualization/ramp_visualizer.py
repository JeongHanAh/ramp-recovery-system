import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict

class RampVisualizer:
    def __init__(self):
        """램프 시각화 도구 초기화"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()
    
    def setup_plot(self):
        """플롯 기본 설정"""
        self.ax.grid(True)
        self.ax.set_xlabel('X Coordinate (m)')
        self.ax.set_ylabel('Y Coordinate (m)')
        self.ax.set_title('Ramp Path Analysis')
        
    def plot_ramp_data(self, data: Dict[str, np.ndarray], 
                      margin: float = 0.1) -> None:
        """램프 데이터 플로팅
        Args:
            data: 플로팅할 데이터
            margin: 그래프 여백 비율
        """
        # 실제 경로
        self.ax.plot(data['x_smooth'], data['y_smooth'], 
                    'b-', label='Ground Truth Path')
        
        # 왜곡된 GPS 경로
        self.ax.plot(data['x_raw'], data['y_raw'], 
                    'r:', label='Distorted GPS Path')
        
        # 축 범위 설정
        x_min, x_max = min(data['x_smooth']), max(data['x_smooth'])
        y_min, y_max = min(data['y_smooth']), max(data['y_smooth'])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.ax.set_xlim(x_min - margin * x_range, 
                        x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, 
                        y_max + margin * y_range)
        
        self.ax.legend()
    
    def plot_correction_animation(self, 
                                gps_positions: List[np.ndarray],
                                corrected_positions: List[np.ndarray],
                                reference_data: Dict[str, np.ndarray],
                                interval: int = 50) -> FuncAnimation:
        """보정 과정 애니메이션
        Args:
            gps_positions: GPS 측정 위치 목록
            corrected_positions: 보정된 위치 목록
            reference_data: 참조 데이터
            interval: 프레임 간격 (ms)
        Returns:
            애니메이션 객체
        """
        # 기준 경로와 GPS 경로 플로팅
        self.plot_ramp_data(reference_data)
        
        # 보정된 경로를 위한 라인
        corrected_line, = self.ax.plot([], [], 'g-', 
                                     label='Corrected Vehicle Path')
        
        # 현재 위치 마커
        position_marker, = self.ax.plot([], [], 'yo', 
                                      markersize=10,
                                      label='Current Vehicle Position')
        
        def init():
            """애니메이션 초기화"""
            corrected_line.set_data([], [])
            position_marker.set_data([], [])
            return corrected_line, position_marker
        
        def update(frame):
            """프레임 업데이트"""
            # 현재까지의 보정된 경로 표시
            x_corrected = [p[0] for p in corrected_positions[:frame+1]]
            y_corrected = [p[1] for p in corrected_positions[:frame+1]]
            corrected_line.set_data(x_corrected, y_corrected)
            
            # 현재 위치 표시
            if frame < len(corrected_positions):
                position_marker.set_data([corrected_positions[frame][0]], 
                                      [corrected_positions[frame][1]])
            
            return corrected_line, position_marker
        
        anim = FuncAnimation(self.fig, update, 
                           frames=len(corrected_positions),
                           init_func=init, 
                           interval=interval,
                           blit=True)
        
        self.ax.legend()
        return anim
    
    def save_plot(self, filepath: str) -> None:
        """플롯 저장"""
        self.fig.savefig(filepath)
    
    def show_plot(self) -> None:
        """플롯 표시"""
        plt.show() 