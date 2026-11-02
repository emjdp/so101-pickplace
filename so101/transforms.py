"""회전행렬과 동차변환. ★ 실습: TODO 를 채운다 ★

확인: pytest tests/test_transforms.py -v
참고: 문서 '2. 순기구학 FK' 1~2절, Modern Robotics 3장

규칙
- 각도는 전부 라디안이다.
- 회전행렬 R 은 (3, 3), 동차변환 T 는 (4, 4) numpy 배열이다.
- T = [[R, p],
       [0, 1]]   p 는 (3,) 위치 벡터
"""

import numpy as np


def rot_x(a: float) -> np.ndarray:
    """x 축으로 a 만큼 돌리는 회전행렬.

    힌트: x 성분은 그대로이고, (y, z) 평면에서 2D 회전이 일어난다.
    """
    raise NotImplementedError("TODO: rot_x")


def rot_y(a: float) -> np.ndarray:
    """y 축으로 a 만큼 돌리는 회전행렬.

    주의: rot_x, rot_z 와 sin 부호 위치가 다르다. z 를 x 쪽으로 돌린다고 생각해 볼 것.
    """
    raise NotImplementedError("TODO: rot_y")


def rot_z(a: float) -> np.ndarray:
    """z 축으로 a 만큼 돌리는 회전행렬. rot_z(pi/2) @ [1, 0, 0] == [0, 1, 0]"""
    raise NotImplementedError("TODO: rot_z")


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF 의 rpy 를 회전행렬로 바꾼다.

    URDF 규약: 고정된 x, y, z 축에 대해 roll → pitch → yaw 순서로 돌린다.
    고정 축 회전은 '나중 회전을 왼쪽에 곱한다'. 즉 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    (같은 결과를 '움직이는 축 z → y → x 순서' 로 읽을 수도 있다. 문서 '서로 설명하기' 6번)
    """
    raise NotImplementedError("TODO: rpy_to_matrix")


def make_transform(R: np.ndarray, p) -> np.ndarray:
    """회전 R (3x3) 과 위치 p (3,) 로 동차변환 T (4x4) 를 만든다."""
    raise NotImplementedError("TODO: make_transform")


def invert_transform(T: np.ndarray) -> np.ndarray:
    """T 의 역변환. np.linalg.inv 를 쓰지 말고 R 의 성질(R^-1 = R^T)로 직접 구한다.

    T = [R p; 0 1] 이면 T^-1 = [R^T  -R^T p; 0 1]  — 왜 그런지 종이에 유도해 볼 것.
    (5주차 카메라-로봇 좌표 변환에서 다시 쓴다)
    """
    raise NotImplementedError("TODO: invert_transform")
