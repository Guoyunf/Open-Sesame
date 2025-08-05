import numpy as np
from utils.lib_math import H_to_xyzrpy, xyzrpy_to_H, euler_to_quaternion_zyx


def test_euler_to_quaternion_zyx_identity():
    quat = euler_to_quaternion_zyx(0.0, 0.0, 0.0)
    assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0])


def test_xyzrpy_roundtrip():
    xyzrpy = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    H = xyzrpy_to_H(xyzrpy, rad=True)
    xyzrpy_back = H_to_xyzrpy(H, rad=True)
    assert np.allclose(xyzrpy, xyzrpy_back)
