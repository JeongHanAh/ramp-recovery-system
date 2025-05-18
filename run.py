import os
import sys

# src 디렉토리를 Python 경로에 추가
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# main 모듈 실행
from main import main

if __name__ == '__main__':
    main() 