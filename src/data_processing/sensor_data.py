import numpy as np
from pyproj import Transformer

class SensorDataProcessor:
    """GPS/IMU 센서 데이터 처리 클래스"""
    
    def __init__(self):
        """
        GPS 좌표를 로컬 평면 좌표계로 변환하기 위한 변환기 초기화
        """
        # WGS84 (EPSG:4326)에서 UTM 좌표계로의 변환기
        self.transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (위도/경도)
            "EPSG:32652",  # UTM Zone 52N (한국 기준)
            always_xy=True
        )
        
        # 초기 위치 저장을 위한 변수
        self.initial_position = None
    
    def process_gps_data(self, longitude, latitude):
        """
        GPS 좌표를 로컬 평면 좌표계로 변환

        매개변수:
            longitude (float): 경도
            latitude (float): 위도

        반환:
            np.ndarray: 로컬 평면 좌표계에서의 위치 (x, y)
        """
        # GPS 좌표를 UTM 좌표로 변환
        x, y = self.transformer.transform(longitude, latitude)
        position = np.array([x, y])
        
        # 초기 위치가 설정되지 않았다면 현재 위치를 초기 위치로 설정
        if self.initial_position is None:
            self.initial_position = position
        
        # 초기 위치를 원점으로 하는 상대 좌표 반환
        return position - self.initial_position
    
    def process_imu_data(self, acceleration, angular_velocity):
        """
        IMU 데이터 처리

        매개변수:
            acceleration (np.ndarray): 3축 가속도 데이터 (x, y, z)
            angular_velocity (np.ndarray): 3축 각속도 데이터 (roll, pitch, yaw)

        반환:
            tuple: (수평면 가속도, 요(yaw) 각속도)
        """
        # 수평면(x-y 평면)에서의 가속도 계산
        horizontal_acc = np.array([acceleration[0], acceleration[1]])
        
        # 요(yaw) 각속도 추출
        yaw_rate = angular_velocity[2]
        
        return horizontal_acc, yaw_rate
    
    def calculate_heading(self, yaw_rate, dt, initial_heading=0.0):
        """
        각속도 적분을 통한 방향 계산

        매개변수:
            yaw_rate (float): 요(yaw) 각속도
            dt (float): 시간 간격
            initial_heading (float): 초기 방향 (라디안)

        반환:
            float: 계산된 방향 (라디안)
        """
        heading = initial_heading + yaw_rate * dt
        
        # 각도를 -π에서 π 사이로 정규화
        heading = np.arctan2(np.sin(heading), np.cos(heading))
        
        return heading 