import json
import os
import numpy as np

# 2차 곡선을 사용하여 램프와 유사한 경로 생성
x = np.linspace(0, 20, 21)
y = 0.1 * x**2

# JSON 데이터 구조 생성
data = {
    "metadata": {
        "description": "고속도로 램프 참조 경로 데이터",
        "date_created": "2024-03-17",
        "coordinate_system": "local_xy"
    },
    "coordinates": {
        "x": x.tolist(),
        "y": y.tolist()
    },
    "parameters": {
        "sampling_rate": 1.0,
        "total_points": len(x)
    }
}

# 저장 경로 확인 및 생성
save_dir = os.path.join("data", "raw", "reference_paths")
os.makedirs(save_dir, exist_ok=True)

# JSON 파일로 저장
save_path = os.path.join(save_dir, "reference_data.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4) 