"""
Data processing package for ramp recovery system
"""

# 순환 임포트 방지를 위해 런타임에 임포트
def get_ramp_geometry():
    from .ramp_geometry import RampGeometry
    return RampGeometry

def get_gps_correction():
    from .gps_correction import GPSCorrector, GPSMeasurement, GPSDistortionSimulator
    return GPSCorrector, GPSMeasurement, GPSDistortionSimulator

def get_ramp_estimator():
    from .ramp_estimator import RampEstimator
    return RampEstimator

def get_gps_3d_error():
    from .gps_3d_error import GPS3DDistortionSimulator
    return GPS3DDistortionSimulator

__all__ = ['get_ramp_geometry', 'get_gps_correction', 'get_ramp_estimator', 'get_gps_3d_error'] 