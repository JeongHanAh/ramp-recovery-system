import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from matplotlib.animation import FuncAnimation

class RampVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        
    def plot_ramp_data(self, data: Dict[str, np.ndarray], title: str = "램프 경로 분석") -> None:
        """램프 데이터 시각화"""
        self.ax.plot(data['x_raw'], data['y_raw'], 'r.', label='원본 데이터', alpha=0.5)
        self.ax.plot(data['x_smooth'], data['y_smooth'], 'b-', label='보정된 경로', linewidth=2)
        
        self.ax.set_title(title)
        self.ax.set_xlabel('X 좌표 (m)')
        self.ax.set_ylabel('Y 좌표 (m)')
        self.ax.legend()
        self.ax.grid(True)
    
    def plot_correction_animation(self, 
                                original_positions: List[np.ndarray],
                                corrected_positions: List[np.ndarray],
                                reference_path: Dict[str, np.ndarray]) -> None:
        """경로 보정 과정 애니메이션"""
        line_ref, = self.ax.plot([], [], 'b-', label='기준 경로')
        line_current, = self.ax.plot([], [], 'r.', label='현재 위치')
        line_corrected, = self.ax.plot([], [], 'g-', label='보정된 경로')
        
        def init():
            self.ax.plot(reference_path['x_smooth'], reference_path['y_smooth'], 'b-', alpha=0.5)
            return line_ref, line_current, line_corrected
        
        def update(frame):
            # 현재까지의 경로 표시
            x_orig = [pos[0] for pos in original_positions[:frame+1]]
            y_orig = [pos[1] for pos in original_positions[:frame+1]]
            x_corr = [pos[0] for pos in corrected_positions[:frame+1]]
            y_corr = [pos[1] for pos in corrected_positions[:frame+1]]
            
            line_current.set_data(x_orig, y_orig)
            line_corrected.set_data(x_corr, y_corr)
            return line_current, line_corrected
        
        anim = FuncAnimation(self.fig, update, frames=len(original_positions),
                           init_func=init, blit=True, interval=100)
        plt.legend()
        return anim
    
    def save_plot(self, filename: str) -> None:
        """플롯 저장"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    def show_plot(self) -> None:
        """플롯 표시"""
        plt.show() 