import numpy as np
from scipy.interpolate import BSpline

class RampCurve:
    """고속도로 램프 곡선을 B-스플라인으로 모델링하는 클래스"""
    
    def __init__(self, control_points, degree=3):
        """
        매개변수:
            control_points (np.ndarray): 제어점 배열 (n x 2)
            degree (int): B-스플라인 차수 (기본값: 3)
        """
        self.control_points = np.array(control_points)
        self.degree = degree
        self._create_bspline()
    
    def _create_bspline(self):
        """B-스플라인 곡선 생성"""
        n_points = len(self.control_points)
        # 균일한 매듭 벡터 생성
        knots = np.linspace(0, 1, n_points - self.degree + 1)
        # 매듭 벡터의 시작과 끝에 degree+1개의 중복 값 추가
        knots = np.concatenate([
            np.zeros(self.degree),
            knots,
            np.ones(self.degree)
        ])
        
        # x좌표와 y좌표에 대한 별도의 B-스플라인 생성
        self.spline_x = BSpline(knots, self.control_points[:, 0], self.degree)
        self.spline_y = BSpline(knots, self.control_points[:, 1], self.degree)
    
    def evaluate(self, t):
        """
        주어진 매개변수 값에서 곡선 위의 점을 계산

        매개변수:
            t (float or np.ndarray): 0과 1 사이의 매개변수 값

        반환:
            np.ndarray: 곡선 위의 점 (x, y)
        """
        return np.column_stack([self.spline_x(t), self.spline_y(t)])
    
    def evaluate_derivative(self, t):
        """
        주어진 매개변수 값에서 곡선의 도함수(접선 벡터) 계산

        매개변수:
            t (float or np.ndarray): 0과 1 사이의 매개변수 값

        반환:
            np.ndarray: 접선 벡터 (dx/dt, dy/dt)
        """
        return np.column_stack([
            self.spline_x.derivative()(t),
            self.spline_y.derivative()(t)
        ])
    
    def find_closest_point(self, point, n_samples=1000):
        """
        주어진 점에서 가장 가까운 곡선 위의 점을 찾음

        매개변수:
            point (np.ndarray): 2D 점 좌표
            n_samples (int): 검색할 샘플 수

        반환:
            tuple: (가장 가까운 점의 매개변수 값, 가장 가까운 점의 좌표)
        """
        t_samples = np.linspace(0, 1, n_samples)
        curve_points = self.evaluate(t_samples)
        
        # 각 샘플 점과 주어진 점 사이의 거리 계산
        distances = np.sum((curve_points - point) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        
        t_closest = t_samples[closest_idx]
        closest_point = self.evaluate(t_closest)
        
        return t_closest, closest_point 