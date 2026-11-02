import numpy as np
import pytest

from so101.transforms import invert_transform, make_transform, rot_x, rot_y, rot_z, rpy_to_matrix

ANGLES = [0.0, 0.3, -1.2, np.pi / 2, np.pi]
X, Y, Z = np.eye(3)


@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
@pytest.mark.parametrize("a", ANGLES)
def test_rotation_matrix_is_proper(rot, a):
    """회전행렬은 R^T R = I, det R = +1 이어야 한다."""
    R = rot(a)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_quarter_turns_follow_right_hand_rule():
    np.testing.assert_allclose(rot_z(np.pi / 2) @ X, Y, atol=1e-9)
    np.testing.assert_allclose(rot_x(np.pi / 2) @ Y, Z, atol=1e-9)
    np.testing.assert_allclose(rot_y(np.pi / 2) @ Z, X, atol=1e-9)


def test_rpy_single_axis():
    np.testing.assert_allclose(rpy_to_matrix(0.4, 0, 0), rot_x(0.4), atol=1e-12)
    np.testing.assert_allclose(rpy_to_matrix(0, 0.4, 0), rot_y(0.4), atol=1e-12)
    np.testing.assert_allclose(rpy_to_matrix(0, 0, 0.4), rot_z(0.4), atol=1e-12)


def test_rpy_order():
    """roll(x) 을 먼저, yaw(z) 를 나중에 적용해야 한다. 곱하는 순서가 틀리면 여기서 걸린다."""
    R = rpy_to_matrix(np.pi / 2, 0, np.pi / 2)
    # x 축: roll 로는 그대로, yaw 로 y 가 된다
    np.testing.assert_allclose(R @ X, Y, atol=1e-9)
    # y 축: roll 로 z 가 되고, yaw 로는 그대로
    np.testing.assert_allclose(R @ Y, Z, atol=1e-9)


def test_make_transform_moves_points():
    R = rot_z(np.pi / 2)
    T = make_transform(R, [1.0, 2.0, 3.0])
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    # 자식 좌표계의 (1, 0, 0) 은 부모 좌표계에서 R @ (1,0,0) + p
    np.testing.assert_allclose(T @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-9)


def test_invert_transform():
    T = make_transform(rpy_to_matrix(0.3, -0.7, 1.1), [0.1, -0.2, 0.5])
    Ti = invert_transform(T)
    np.testing.assert_allclose(Ti @ T, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(Ti, np.linalg.inv(T), atol=1e-9)
