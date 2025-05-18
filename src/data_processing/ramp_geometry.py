import numpy as np
import pandas as pd
import json
from scipy.interpolate import CubicSpline
from scipy import ndimage
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import random

class RampGeometry:
    def __init__(self, data_path: str):
        """Initialize ramp geometry from road data
        Args:
            data_path: Path to road geometry data file (.json)
        """
        self.data_path = data_path
        self.load_json_data()
        self.create_spline_model()
        
    def load_json_data(self):
        """Load and process ramp data from JSON file"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        # Extract coordinates
        coords = data['coordinates']
        self.x = np.array(coords['x'])
        self.y = np.array(coords['y'])
        self.z = np.array(coords['z'])
        
        # Store design parameters
        self.metadata = data['metadata']
        self.parameters = data['parameters']
        self.design_constraints = data['design_constraints']
        
        # Normalize coordinates for better visualization
        self.normalize_coordinates()
        
    def normalize_coordinates(self):
        """Normalize coordinates to improve visualization"""
        # Center the coordinates
        self.x -= np.mean(self.x)
        self.y -= np.mean(self.y)
        
        # Scale to reasonable range (e.g., -50 to 50)
        max_range = max(np.ptp(self.x), np.ptp(self.y))
        scale_factor = 100.0 / max_range
        
        self.x *= scale_factor
        self.y *= scale_factor
        self.z *= scale_factor * 0.3  # Scale height less to maintain realistic proportions
        
    def create_spline_model(self):
        """Create smooth spline model of the ramp"""
        # Path length parameterization
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dz = np.diff(self.z)
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        t = np.concatenate(([0], np.cumsum(distances)))
        t = t / t[-1]  # Normalize to [0, 1]
        
        # Fit cubic splines with natural boundary conditions
        self.spline_x = CubicSpline(t, self.x, bc_type='natural')
        self.spline_y = CubicSpline(t, self.y, bc_type='natural')
        self.spline_z = CubicSpline(t, self.z, bc_type='natural')
        
    def get_ramp_points(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get points along the smoothed ramp curve
        Args:
            num_points: Number of points to generate
        Returns:
            Tuple of (x, y, z) coordinates arrays
        """
        t = np.linspace(0, 1, num_points)
        return (self.spline_x(t), self.spline_y(t), self.spline_z(t))
        
    def plot_ramp(self, ax: Optional[plt.Axes] = None,
                 raw_gps: Optional[np.ndarray] = None,
                 corrected_gps: Optional[np.ndarray] = None) -> plt.Axes:
        """Visualize ramp structure with optional GPS data
        Args:
            ax: Matplotlib axis (created if None)
            raw_gps: Raw GPS measurements (Nx3 array)
            corrected_gps: Corrected GPS positions (Nx3 array)
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        # Plot ramp structure (thinner line)
        x, y, z = self.get_ramp_points(300)
        ax.plot(x, y, z, 'b-', label='Ramp Structure', linewidth=1.0, alpha=0.8)
        
        if raw_gps is not None:
            # Plot raw GPS measurements as a line
            ax.plot(raw_gps[:, 0], raw_gps[:, 1], raw_gps[:, 2],
                   'r-', alpha=0.7, label='Raw GPS', linewidth=1.2)
            
        if corrected_gps is not None:
            # Plot corrected GPS path
            ax.plot(corrected_gps[:, 0], corrected_gps[:, 1], corrected_gps[:, 2],
                   'g-', alpha=0.8, label='Corrected GPS', linewidth=1.2)
            
        # Set axis labels and title
        ax.set_xlabel('X Position (m)', labelpad=10)
        ax.set_ylabel('Y Position (m)', labelpad=10)
        ax.set_zlabel('Elevation (m)', labelpad=10)
        ax.set_title('3D Ramp Structure and GPS Data', pad=20)
        
        # Adjust view angle for better visibility
        ax.view_init(elev=30, azim=45)
        
        # Set axis ranges with margin
        margin = 5
        ax.set_xlim([np.min(x) - margin, np.max(x) + margin])
        ax.set_ylim([np.min(y) - margin, np.max(y) + margin])
        ax.set_zlim([0, np.max(z) + margin])
        
        # Adjust aspect ratio to prevent distortion
        ax.set_box_aspect([1, 1, 0.4])
        
        # Customize grid and legend
        ax.grid(True, alpha=0.2, linestyle=':')  # 더 얇은 그리드
        ax.legend(loc='upper right', framealpha=0.9)
        
        return ax

    def _estimate_elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate elevation based on road design standards
        Args:
            x: x coordinates array
            y: y coordinates array
        Returns:
            Estimated elevation array
        """
        # Calculate path length
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
        path_length = cumulative_dist[-1]
        
        # Set ramp parameters
        base_slope = 0.06  # 6% grade
        transition_length = path_length * 0.2  # 20% transition zones
        
        # Calculate elevation profile
        z = np.zeros_like(cumulative_dist)
        for i, d in enumerate(cumulative_dist):
            if d <= transition_length:
                # Entry transition using sine curve
                progress = d / transition_length
                slope = base_slope * np.sin(progress * np.pi/2)
                z[i] = d * slope
            elif d >= (path_length - transition_length):
                # Exit transition using sine curve
                progress = (path_length - d) / transition_length
                slope = base_slope * np.sin(progress * np.pi/2)
                if i > 0:
                    z[i] = z[i-1] + distances[i-1] * slope
            else:
                # Constant grade section
                if i > 0:
                    z[i] = z[i-1] + distances[i-1] * base_slope
        
        return z

    def calculate_curvature(self, x: np.ndarray, y: np.ndarray, window_size: int = 5):
        """이동 평균을 사용한 부드러운 곡률 계산"""
        if len(x) < window_size:
            return np.zeros_like(x)
            
        # 이동 평균으로 노이즈 제거
        x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        
        # 미분 계산
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # 곡률 계산
        curvature = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy)**1.5
        return curvature
        
    def identify_ramp_segments(self, curvature: np.ndarray, min_segment_length: int = 10):
        """적응형 임계값을 사용한 램프 구간 식별"""
        # JSON 데이터는 이미 램프 구간이 정의되어 있으므로 사용하지 않음
        pass
        
    def extract_ramp_segments(self):
        """JSON 데이터는 이미 램프 구간이 정의되어 있으므로 사용하지 않음"""
        pass
            
    def fit_spline_to_segment(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """곡선 길이 기반 매개변수화를 사용한 3D 스플라인 피팅"""
        # 곡선 길이 기반 매개변수화
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        t = cumulative_length / cumulative_length[-1]
        
        # 3차 스플라인 피팅 (자연 경계 조건 사용)
        spline_x = CubicSpline(t, x, bc_type='natural')
        spline_y = CubicSpline(t, y, bc_type='natural')
        spline_z = CubicSpline(t, z, bc_type='natural')
        
        return spline_x, spline_y, spline_z
    
    def project_point(self, point: np.ndarray, ramp_index: int = 0) -> Tuple[float, np.ndarray]:
        """주어진 3D 점을 가장 가까운 램프 곡선에 투영"""
        if ramp_index >= len(self.ramp_splines):
            raise ValueError("Invalid ramp index")
            
        spline_x, spline_y, spline_z = self.ramp_splines[ramp_index]
        t_samples = np.linspace(0, 1, 200)
        
        # 각 t에서의 곡선 위의 점과 주어진 점 사이의 거리 계산
        distances = []
        curve_points = []
        for t in t_samples:
            curve_point = np.array([spline_x(t), spline_y(t), spline_z(t)])
            curve_points.append(curve_point)
            distance = np.linalg.norm(curve_point - point)
            distances.append(distance)
        
        # 최소 거리를 갖는 t 값 찾기
        min_idx = np.argmin(distances)
        t_min = t_samples[min_idx]
        projected_point = np.array(curve_points[min_idx])
        
        return t_min, projected_point
    
    def get_tangent(self, t: float, ramp_index: int = 0) -> np.ndarray:
        """주어진 매개변수 t에서의 3D 접선 벡터 계산"""
        if ramp_index >= len(self.ramp_splines):
            raise ValueError("Invalid ramp index")
            
        spline_x, spline_y, spline_z = self.ramp_splines[ramp_index]
        dx = spline_x.derivative()(t)
        dy = spline_y.derivative()(t)
        dz = spline_z.derivative()(t)
        tangent = np.array([dx, dy, dz])
        return tangent / np.linalg.norm(tangent)
    
    def get_curvature(self, t: float, ramp_index: int = 0) -> float:
        """주어진 매개변수 t에서의 3D 곡률 계산"""
        if ramp_index >= len(self.ramp_splines):
            raise ValueError("Invalid ramp index")
            
        spline_x, spline_y, spline_z = self.ramp_splines[ramp_index]
        dx = spline_x.derivative()(t)
        dy = spline_y.derivative()(t)
        dz = spline_z.derivative()(t)
        d2x = spline_x.derivative(2)(t)
        d2y = spline_y.derivative(2)(t)
        d2z = spline_z.derivative(2)(t)
        
        # 3D 곡률 계산
        velocity = np.array([dx, dy, dz])
        acceleration = np.array([d2x, d2y, d2z])
        cross = np.cross(velocity, acceleration)
        return np.linalg.norm(cross) / np.linalg.norm(velocity)**3
    
    def plot_ramps(self, ax: Optional[plt.Axes] = None):
        """3D 램프 곡선 시각화"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        # 전체 도로 데이터 표시 (회색)
        ax.plot(self.x, self.y, self.z, 'gray', alpha=0.5, label='전체 도로')
            
        # 램프 구간 강조 표시 (파란색)
        for x, y, z in self.get_ramp_points(num_points=200):
            ax.plot(x, y, z, 'b-', linewidth=2)
            
        # 축 범위 및 비율 설정
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y) - np.min(self.y)
        z_range = np.max(self.z) - np.min(self.z)
        
        # 고도 스케일 조정
        ax.set_box_aspect([x_range/max(x_range, y_range), 
                          y_range/max(x_range, y_range),
                          z_range/max(x_range, y_range) * 2.0])  # 고도 강조
        
        # 보기 각도 최적화
        ax.view_init(elev=25, azim=45)
        
        # 그리드 및 레이블 설정
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('고도 (m)', fontsize=12, labelpad=10)
        ax.set_title('3D 램프 형상', pad=20, fontsize=14)
        
        # 축 눈금 개선
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # 여백 추가
        margin = max(x_range, y_range, z_range) * 0.1
        ax.set_xlim([np.min(self.x) - margin, np.max(self.x) + margin])
        ax.set_ylim([np.min(self.y) - margin, np.max(self.y) + margin])
        ax.set_zlim([0, np.max(self.z) + margin])
        
        return ax 