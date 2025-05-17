import numpy as np
from ..curve_modeling.bspline import RampCurve

class PathRecovery:
    """실시간 경로 복구 알고리즘 클래스"""
    
    def __init__(self, ramp_curve: RampCurve, k_tangent=0.5, k_normal=0.5):
        """
        매개변수:
            ramp_curve (RampCurve): 목표 램프 곡선
            k_tangent (float): 접선 방향 보정 가중치
            k_normal (float): 법선 방향 보정 가중치
        """
        self.ramp_curve = ramp_curve
        self.k_tangent = k_tangent
        self.k_normal = k_normal
    
    def calculate_correction_vector(self, current_position):
        """
        현재 위치에서의 보정 벡터 계산

        매개변수:
            current_position (np.ndarray): 현재 위치 (x, y)

        반환:
            np.ndarray: 보정 벡터 (dx, dy)
        """
        # 가장 가까운 곡선 위의 점 찾기
        t_closest, closest_point = self.ramp_curve.find_closest_point(current_position)
        
        # 오차 벡터 계산 (현재 위치에서 가장 가까운 곡선 위의 점까지)
        error_vector = closest_point - current_position
        
        # 접선 벡터 계산 및 정규화
        tangent = self.ramp_curve.evaluate_derivative(t_closest)
        tangent = tangent / np.linalg.norm(tangent)
        
        # 법선 벡터 계산 (접선 벡터를 90도 회전)
        normal = np.array([-tangent[1], tangent[0]])
        
        # 오차 벡터를 접선과 법선 방향으로 분해
        error_tangent = np.dot(error_vector, tangent) * tangent
        error_normal = np.dot(error_vector, normal) * normal
        
        # 보정 벡터 계산
        correction_vector = (
            self.k_tangent * error_tangent +
            self.k_normal * error_normal
        )
        
        return correction_vector
    
    def get_steering_angle(self, current_position, vehicle_heading):
        """
        조향각 계산

        매개변수:
            current_position (np.ndarray): 현재 위치 (x, y)
            vehicle_heading (float): 차량의 현재 방향 (라디안)

        반환:
            float: 권장 조향각 (라디안)
        """
        correction = self.calculate_correction_vector(current_position)
        
        # 보정 벡터의 방향 계산
        target_heading = np.arctan2(correction[1], correction[0])
        
        # 현재 방향과의 차이 계산
        heading_error = target_heading - vehicle_heading
        
        # 각도를 -π에서 π 사이로 정규화
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        return heading_error 