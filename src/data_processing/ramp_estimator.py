"""Real-time ramp structure estimation module"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
import json
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .ramp_geometry import RampGeometry

class RampEstimator:
    def __init__(self, reference_data_path: str):
        """Initialize ramp structure estimator
        Args:
            reference_data_path: Path to reference path data file
        """
        self.ramp_geometry = RampGeometry(reference_data_path)
        
        # Initialize KD-tree for efficient nearest point queries
        self._initialize_kdtree()
        
    def _initialize_kdtree(self, num_points: int = 2000):
        """Initialize KD-tree with dense points along the ramp
        Args:
            num_points: Number of points to use for KD-tree
        """
        t = np.linspace(0, 1, num_points)
        x = self.ramp_geometry.spline_x(t)
        y = self.ramp_geometry.spline_y(t)
        z = self.ramp_geometry.spline_z(t)
        
        self.reference_points = np.column_stack([x, y, z])
        self.reference_t = t
        self.kdtree = cKDTree(self.reference_points)
        
    def get_position(self, t: float) -> np.ndarray:
        """Get 3D position at given parameter t
        Args:
            t: Path parameter (0~1)
        Returns:
            3D position [x, y, z]
        """
        t = np.clip(t, 0, 1)
        return np.array([
            float(self.ramp_geometry.spline_x(t)),
            float(self.ramp_geometry.spline_y(t)),
            float(self.ramp_geometry.spline_z(t))
        ])
        
    def get_tangent(self, t: float) -> np.ndarray:
        """Get unit tangent vector at given parameter t
        Args:
            t: Path parameter (0~1)
        Returns:
            Unit tangent vector [dx, dy, dz]
        """
        t = np.clip(t, 0, 1)
        tangent = np.array([
            float(self.ramp_geometry.spline_x.derivative()(t)),
            float(self.ramp_geometry.spline_y.derivative()(t)),
            float(self.ramp_geometry.spline_z.derivative()(t))
        ])
        return tangent / np.linalg.norm(tangent)
        
    def get_curvature(self, t: float) -> float:
        """Get curvature at given parameter t
        Args:
            t: Path parameter (0~1)
        Returns:
            Curvature value
        """
        t = np.clip(t, 0, 1)
        dx = self.ramp_geometry.spline_x.derivative()
        dy = self.ramp_geometry.spline_y.derivative()
        dz = self.ramp_geometry.spline_z.derivative()
        
        d2x = self.ramp_geometry.spline_x.derivative(2)
        d2y = self.ramp_geometry.spline_y.derivative(2)
        d2z = self.ramp_geometry.spline_z.derivative(2)
        
        velocity = np.array([dx(t), dy(t), dz(t)])
        acceleration = np.array([d2x(t), d2y(t), d2z(t)])
        
        cross = np.cross(velocity, acceleration)
        return np.linalg.norm(cross) / np.linalg.norm(velocity)**3
        
    def get_closest_point(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Find closest point on ramp curve to given 3D point
        Args:
            point: 3D point [x, y, z]
        Returns:
            (parameter t, closest point coordinates)
        """
        # Scale point for distance computation
        scaled_point = point.copy()
        scaled_point[2] *= 1.5  # Give more weight to elevation differences
        
        # Find nearest point
        _, idx = self.kdtree.query(scaled_point)
        return self.reference_t[idx], self.reference_points[idx]
        
    def get_path_points(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get points along the ramp curve
        Args:
            num_points: Number of points to generate
        Returns:
            (x, y, z) coordinate arrays
        """
        return self.ramp_geometry.get_ramp_points(num_points)
        
    def plot_results(self, raw_gps: np.ndarray, corrected_gps: np.ndarray) -> plt.Axes:
        """Plot ramp structure with raw and corrected GPS data
        Args:
            raw_gps: Raw GPS measurements (Nx3 array)
            corrected_gps: Corrected GPS positions (Nx3 array)
        Returns:
            Matplotlib axis object
        """
        return self.ramp_geometry.plot_ramp(
            raw_gps=raw_gps,
            corrected_gps=corrected_gps
        ) 