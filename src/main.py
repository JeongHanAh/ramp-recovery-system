# 전체 실행 로직
import numpy as np
import sys
import os

# 프로젝트 루트 디렉토리를 절대 경로로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from spline_curve import fit_spline
from recovery_algo import recovery_step

def main():
    # 예제 데이터
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 0, -1, 0, 1]
    p = np.array([1.5, 0.5])  # 초기 위치

    r_func = fit_spline(x, y)

    # 보정 반복
    for step in range(10):
        p = recovery_step(p, r_func, t=step, alpha=0.7, beta=0.3, eta=1.0, delta_t=0.1)
        print(f"Step {step}: 위치 = {p}")

if __name__ == '__main__':
    main()