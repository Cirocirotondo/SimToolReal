import numpy as np
import cv2

import copy

def scalar_last_to_scalar_first(pose):
    assert len(pose) == 7

    tmp = copy.deepcopy(pose)

    [x, y, z, w] = tmp[3:]
    tmp[3:] = [w, x, y, z]
    # pose[3] = pose[6]
    # pose[4] = pose[3]
    # pose[5] = pose[4]
    # pose[6] = pose[5]

    return tmp


def scalar_first_to_scalar_last(pose):
    assert len(pose) == 7
    tmp = copy.deepcopy(pose)

    [w, x, y, z] = tmp[3:]
    tmp[3:] = [x, y, z, w]

    return tmp

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix

    Returns:
    numpy.ndarray: Quaternion in format [w, x, y, z]
    """
    # Ensure R is a valid rotation matrix
    if R.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3")

    # Check for proper orthogonal matrix
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError("Input matrix must have determinant 1")

    # Computing trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        # If trace is positive
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # If R[0,0] is the largest diagonal entry
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        # If R[1,1] is the largest diagonal entry
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        # If R[2,2] is the largest diagonal entry
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    # Return quaternion as [w, x, y, z]
    return np.array([w, x, y, z])


def pose_to_homogeneous(rvec, tvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Construct the homogeneous transformation matrix
    T = np.eye(4)  # Initialize as identity
    T[:3, :3] = R  # Set rotation
    T[:3, 3] = tvec.flatten()  # Set translation

    return T


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

def seven_d_to_homogeneous(pose):
    # Convert rotation vector to rotation matrix
    R = quat_to_rotmat(pose[3:])

    # Construct the homogeneous transformation matrix
    T = np.eye(4)  # Initialize as identity
    T[:3, :3] = R  # Set rotation
    T[:3, 3] = pose[:3]  # Set translation

    return T


def load_camera_calibration(filepath):
    # rvec = np.load(rvec_file)
    # pose = pose_to_homogeneous(rvec, tvec)

    pose = np.load(filepath)
    return pose


def translation_from_homogenous(T):
    return T[:3, 3]


def rotation_from_homogenous(T):
    return T[:3, :3]
