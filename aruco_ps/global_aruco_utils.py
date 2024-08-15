import numpy as np

def check_nan(x):
    if np.isnan(x).any() or np.isinf(x).any() or np.isneginf(x).any():
        return True
    return False

def check_nan_list(x):
    for i in x:
        if check_nan(i):
            return True
    return False

def check_nan_np(x):
    if np.isnan(x).any() or np.isinf(x).any() or np.isneginf(x).any():
        return True
    return False


def cveul_aaeeul(cveul):
    aaeeul = np.zeros(3)
    aaeeul[0] = cveul[2]
    aaeeul[1] = cveul[0]
    aaeeul[2] = cveul[1]
    return aaeeul

def eul2rotm(eul):
    """
    Convert Euler angles to rotation matrix.

    Parameters:
    - eul: Euler angles (roll, pitch, yaw) in radians

    Returns:
    - R: Rotation matrix ZYX
    """
    roll, pitch, yaw = eul[0], eul[1], eul[2]
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R_zyx = np.dot(R_z, np.dot(R_y, R_x))
    return R_zyx

def cvXYZ_aaeXYZ(cvXYZ):
    aaeXYZ = np.zeros(3)
    try:
        aaeXYZ[0] = -cvXYZ[2]
        aaeXYZ[1] = cvXYZ[0]
        aaeXYZ[2] = -cvXYZ[1]
    except:
        aaeXYZ = np.array([0., 0., 0.])
    return aaeXYZ

def get_aae_T(eul_aa, XYZ_aa):
    R = eul2rotm(eul_aa)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = XYZ_aa
    return T