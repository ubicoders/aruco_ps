import numpy as np

def triangulate_point(K1, K2, pL, pR, baseline):
    """
    Triangulate a 3D point from corresponding points in stereo images.

    Parameters:
    - K1: Intrinsic matrix of the left camera (3x3)
    - K2: Intrinsic matrix of the right camera (3x3)
    - pL: Point in the left image (tuple or list of (x_l, y_l))
    - pR: Point in the right image (tuple or list of (x_r, y_r))
    - baseline: Distance between the two cameras in the stereo setup (in meters)

    Returns:
    - X, Y, Z: 3D coordinates of the point in space
    """

    # Convert points to homogeneous coordinates
    pL_hom = np.array([pL[0], pL[1], 1.0])
    pR_hom = np.array([pR[0], pR[1], 1.0])

    # Convert image points to normalized camera coordinates
    P_L = np.linalg.inv(K1).dot(pL_hom)
    P_R = np.linalg.inv(K2).dot(pR_hom)

    # Projection matrices
    P_L_matrix = np.hstack((np.eye(3), np.zeros((3, 1))))
    P_R_matrix = np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T))

    # Construct the matrix A for solving AX = 0
    A = np.array([
        P_L[0] * P_L_matrix[2, :] - P_L_matrix[0, :],
        P_L[1] * P_L_matrix[2, :] - P_L_matrix[1, :],
        P_R[0] * P_R_matrix[2, :] - P_R_matrix[0, :],
        P_R[1] * P_R_matrix[2, :] - P_R_matrix[1, :]
    ])

    # Solve for X using SVD (AX = 0)
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    
    # Convert homogeneous coordinates to 3D Cartesian coordinates
    X = X_homogeneous[0] / X_homogeneous[3]
    Y = X_homogeneous[1] / X_homogeneous[3]
    Z = X_homogeneous[2] / X_homogeneous[3]

    return X, Y, Z

