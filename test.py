from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as R
import random
import math
from scipy.linalg import svd
from skimage.metrics import structural_similarity as ssim

#PLOTING FUNCTIONS

#--------------------------------------------------------------------------------------------------------------------------

def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)


def plot_camera(ax, T, label, color='r'):
    """Plots a camera at the specified transformation matrix."""
    camera_pos = T[:3, 3]  # Extract camera position
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color=color, marker='o')
    ax.text(camera_pos[0], camera_pos[1], camera_pos[2], label, color=color)

def onclick(event, img1, img2, F_21, ax1, ax2):
    """
    Callback function that handles the click event in either Image 1 or Image 2 and plots the epipolar lines.
    - event: Click event from matplotlib
    - img1: Image 1 array (for visualization)
    - img2: Image 2 array (for visualization)
    - F_21: Fundamental matrix (3x3)
    - ax1: Axis handle for Image 1
    - ax2: Axis handle for Image 2
    """
    if event.inaxes:
        if event.inaxes == ax1:  # If clicked in Image 1
            x1 = np.array([event.xdata, event.ydata])
            x1_homogeneous = np.hstack((x1, 1))
            l2 = compute_epipolar_line(F_21, x1_homogeneous)

            # Plot epipolar line in Image 2
            ax2.imshow(img2, cmap='gray')  # Reset the image
            plot_epipolar_line_in_image(ax2, l2, img2.shape)
            plt.draw()

        elif event.inaxes == ax2:  # If clicked in Image 2
            x2 = np.array([event.xdata, event.ydata])
            x2_homogeneous = np.hstack((x2, 1))
            l1 = compute_epipolar_line(F_21.T, x2_homogeneous)

            # Plot epipolar line in Image 1
            ax1.imshow(img1, cmap='gray')  # Reset the image
            plot_epipolar_line_in_image(ax1, l1, img1.shape)
            plt.draw()

def plot_epipolar_line_in_image(ax, line, img_shape):
    """
    Plots an epipolar line on the image.
    - ax: Axis handle to plot on.
    - line: Epipolar line (in homogeneous coordinates).
    - img_shape: Shape of the image to set boundaries for the line.
    """
    # Get the image dimensions
    h, w = img_shape[:2]

    # Epipolar line is of the form ax + by + c = 0
    a, b, c = line

    # Compute points where the line intersects the image borders
    x0, x1 = 0, w  # x coordinates (left and right image borders)
    y0 = -(a * x0 + c) / b if b != 0 else float('inf')  # y at left border
    y1 = -(a * x1 + c) / b if b != 0 else float('inf')  # y at right border

    # Clip the line to only plot within the image frame
    ax.plot([x0, x1], [y0, y1], 'g-', linewidth=2)

#-------------------------------------------------------------------------------------------------------------------------------------

#EXTRACTION FUNCTIONS

#-------------------------------------------------------------------------------------------------------------------------------------

def extract_keypoints_descriptors(npz_path) -> np.array:
    # Load the matches from the npz file
    npz = np.load(npz_path)
    
    # Extract keypoints and descriptors for the two images involved in the match
    kpts0 = npz['keypoints0']  # Keypoints from image 1
    kpts1 = npz['keypoints1']  # Keypoints from image 2
    dsc0 = npz['descriptors0']  # Descriptors from image 1
    dsc1 = npz['descriptors1']  # Descriptors from image 2
    matches = npz['matches']  # The match indices

    # Return the valid matches and the corresponding keypoints
    return kpts0, kpts1, matches, dsc0, dsc1

def extract_matches(kp1, kp2, matches):
    kp1_match = []
    kp2_match = []
    
    for i in range(len(matches)):
        if(matches[i]!= -1):
            kp1_match.append(kp1[i])
            kp2_match.append(kp2[matches[i]])
    
    return kp1_match, kp2_match

def get_common_keypoints(kp1_sets) -> np.array:
    # Assuming kp1_sets is a list of numpy arrays of keypoint coordinates
    # We will extract the points (coordinates) and find common ones
    common_points = set(map(tuple, kp1_sets[0]))  # Convert the first set of points to a set of tuples
    for kp1_set in kp1_sets[1:]:
        current_points = set(map(tuple, kp1_set))  # Convert the current set of points to a set of tuples
        common_points &= current_points  # Get intersection of points
    return [np.array(pt) for pt in common_points]

def get_common_kp2(kp1, kp2, common_kp1) -> np.array:
    common_kp2 = []
    
    # Iterate through common_kp1 (which is a set of keypoints)
    for kp1_pt in common_kp1:
        # Check if the keypoint exists in kp1
        for i, kp1_match in enumerate(kp1):
            # Compare keypoints based on coordinates (or other criteria)
            if tuple(kp1_match) == tuple(kp1_pt):
                common_kp2.append(kp2[i])
                break  # Found the corresponding match, so break inner loop
    
    return common_kp2

def warp_points_back_to_original(distorted_points, original_shape, distorted_shape):
    """
    Warp points from the distorted image back to the original image using a homography.
    
    :param distorted_points: List of points in the distorted image (Nx2 or Nx1 array)
    :param original_shape: Shape of the original image (height, width)
    :param distorted_shape: Shape of the distorted image (height, width)
    
    :return: Points mapped back to the original image.
    """
    # Normalize points from distorted image
    distorted_points = np.array(distorted_points, dtype=np.float32)
    
    # Create scaling transformation matrices based on image sizes
    scale_x = original_shape[1] / distorted_shape[1]
    scale_y = original_shape[0] / distorted_shape[0]
    
    # Create a scaling matrix for transforming points
    scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    
    # Apply scaling to distorted points
    scaled_points = cv2.transform(np.array([distorted_points]), scaling_matrix)[0]
    
    return scaled_points

#--------------------------------------------------------------------------------------------------------------------

#OPERATION FUNCTIONS

#------------------------------------------------------------------------------------------------------------------------

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def ensamble_P(K, T) -> np.array:
    """
    Ensamble the P matrix multypling K matrix and T matrix.
    """
    ProjMatrix= np.hstack((np.eye(3), np.zeros((3, 1))))
    P = K @ ProjMatrix @ T
    return P

def compute_rmse(observed_points, reprojected_points):
    """
    Calculate the RMSE between observed points and reprojected points.
    
    Parameters:
        observed_points (np.array): Array of shape (N, 2) containing the observed 2D points.
        reprojected_points (np.array): Array of shape (N, 2) containing the reprojected 2D points.
        
    Returns:
        float: The RMSE value.
    """

    squared_diffs = np.sum((observed_points - reprojected_points) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diffs))
    return rmse

def project_points(P, X_w) -> np.array:
    """
    Projects 3D world points into the image plane using projection matrix P.
    - P: Projection matrix (3x4)
    - X_w: 3D points in world coordinates (4xN, homogeneous coordinates)
    Returns 2D projected points.
    """
    # Convert 3D points to homogeneous coordinates (add a row of ones)
    X_w_homogeneous = np.vstack((X_w, np.ones((1, X_w.shape[1]))))
    
    # Project points using the projection matrix
    x_projected_homogeneous = P @ X_w_homogeneous
    
    # Convert from homogeneous to 2D (normalize by the last row)
    x_projected = x_projected_homogeneous[:2] / x_projected_homogeneous[2]
    
    return x_projected

def compute_epipolar_line(F, x1):
    """
    Computes the epipolar line l2 in image 2 corresponding to point x1 in image 1.
    - F: Fundamental matrix (3x3)
    - x1: 2D point in image 1 (in homogeneous coordinates)
    
    Returns the normalized epipolar line l2 (in homogeneous coordinates) in image 2.
    """
    # Ensure the point is in homogeneous coordinates
    if x1.shape[0] == 2:
        x1 = np.hstack((x1, 1))

    # Compute the epipolar line in image 2
    l2 = F @ x1

    # Normalize the epipolar line to avoid scaling issues
    l2 = l2 / np.sqrt(l2[0]**2 + l2[1]**2)
    return l2

def fundamental_To_Essential(F, K1):
    return  K1.T @ F @ K1

def triangulate_points(P1, P2, points1, points2) -> np.array:
    """
    Triangulate 3D points from two views using SVD.

    Parameters:
    P1 (np.ndarray): 3x4 projection matrix for the first camera.
    P2 (np.ndarray): 3x4 projection matrix for the second camera.
    points1 (np.ndarray): Nx2 array of 2D points in the first image.
    points2 (np.ndarray): Nx2 array of 2D points in the second image.

    Returns:
    np.ndarray: Nx3 array of triangulated 3D points.
    """
    
    num_points = points1.shape[0]
    points_3d = np.zeros((num_points, 3))
    
    for i in range(num_points):
        # Get the 2D points from both views
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Formulate the A matrix for each point pair
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]
        
        # Perform SVD on A
        _, _, Vt = np.linalg.svd(A)
        
        # The solution is the last column of V (or the last row of V transposed)
        X = Vt[-1]
        
        # Normalize the homogeneous coordinate
        X = X / X[-1]
        
        # Store the 3D point
        points_3d[i] = X[:3]
    
    return np.array(points_3d)

def ransac_fundamental_matrix(kp1, kp2, n_iterations=1000, threshold=1.0):
    """
    RANSAC algorithm for robustly estimating the fundamental matrix from SuperGlue matches.
    
    - matches: A list of tuples (matches_ref_*) containing matched points between two images.
    - n_iterations: Number of RANSAC iterations.
    - threshold: Distance threshold for inliers.
    
    Returns:
    - best_F: The best estimated fundamental matrix.
    - best_inliers: List of matches that are considered inliers for the best F.
    """
    
    pts1 = kp1
    pts2 = kp2
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    if len(pts1) < 8:  # RANSAC needs at least 8 points to estimate the fundamental matrix
        raise ValueError("Not enough points for RANSAC")

    best_F = None
    n_best_inliers = 0

    # RANSAC algorithm for fundamental matrix calculation
    for _ in range(n_iterations):
        # Randomly sample 8 points (index sampling) from the flattened points
        sample_indices = random.sample(range(len(pts1)), 8)
        sampled_pts1 = pts1[sample_indices]
        sampled_pts2 = pts2[sample_indices]

        # Compute the fundamental matrix using these 8 points
        F = compute_fundamental_matrix(sampled_pts1, sampled_pts2)

        # Compute transfer error for all points
        errors = compute_transfer_error_Fund(F, pts1, pts2)

        # Identify inliers based on the error threshold
        inliers = [i for i, e in enumerate(errors) if e < threshold]

        # Update if this set has more inliers than the previous best
        if len(inliers) > n_best_inliers:
            n_best_inliers = len(inliers)
            best_F = F

    return best_F

def compute_fundamental_matrix(pts1, pts2):
    """
    Compute the fundamental matrix using the 8-point algorithm.
    """
    A = np.zeros((len(pts1), 9))
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        A[i] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]
    
    _, _, Vh = np.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F (by setting the smallest singular value to zero)
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh

    return F / F[2, 2]

def compute_transfer_error_Fund(F, pts1, pts2):
    """
    Compute the transfer error (distance from point to corresponding epipolar line).
    """
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  # Convert pts1 to homogeneous coordinates
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # Convert pts2 to homogeneous coordinates

    # Epipolar lines in the second image for points in the first image: l2 = F * pts1_h
    lines2 = F @ pts1_h.T
    # Epipolar lines in the first image for points in the second image: l1 = F.T * pts2_h
    lines1 = F.T @ pts2_h.T

    # Compute distances of points from the corresponding epipolar lines
    num = np.abs(np.sum(lines2.T * pts2_h, axis=1))
    denom = np.sqrt(lines2[0]**2 + lines2[1]**2)
    errors = num / denom

    return errors

def decompose_essential_matrix(E):
    """
    Decomposes the essential matrix into possible rotations and translations.
    """
    #Defina W

    W= np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(E)

    # Ensuring a proper rotation matrix
    R_minus = U @ W.T @ Vt
    R_plus = U @ W @ Vt

    if np.linalg.det(R_minus< -0.5):
        R_minus = -R_minus

    if np.linalg.det(R_plus< -0.5):
        R_plus = -R_plus        

    # Translation vector (the last column of U)
    t = U[:, 2]

    poses = [
        (R_plus, t),
        (R_plus, -t),
        (R_minus, t),
        (R_minus, -t)
    ]
    
    return poses

def best_pose_estimation(T_21_candidates, triangulated_points, P1, K_c):
    best_count = -1
    best_pose = None
    best_points = np.array([])
    for i, T_21 in enumerate(T_21_candidates):
        # Construct the second camera's projection matrix
        #P2 = ensamble_P(K_c, np.linalg.inv(T_21))
        P2 = ensamble_P(K_c,T_21)
        # Count the number of points in front of both cameras
        count = count_points_in_front_of_cameras(triangulated_points[i], T_21, P1, P2)

        print(f"Candidate {i+1}: Points in front of both cameras = {count}")

        # Update best pose if current count is greater
        if count >= best_count:
            best_count = count
            best_pose = T_21
            best_points = triangulated_points[i]

    return best_pose, best_points

def count_points_in_front_of_cameras(points_3d, T_21, P1, P2) -> np.array:
    # Convert to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # # Project points onto the image planes of both cameras
    # points_camera1 = P1 @ points_3d_homogeneous.T  # Shape (3, N)
    # #points_camera2 = P2 @ (T_21 @ points_3d_homogeneous.T)  # Shape (3, N)
    # points_camera2 = P2 @ (points_3d_homogeneous.T)

    points_camera1 = project_points(P1, points_3d_homogeneous.T)
    points_camera2 = project_points(P2, points_3d_homogeneous.T)
    
    # Check if points are in front of both cameras (z > 0)
    in_front_camera1 = 0
    in_front_camera2 = 0
    
    for i in range(points_camera1.shape[1]):
        if points_camera1[2, i] > 0:
            in_front_camera1 += 1
        if points_camera2[2, i] > 0:
            in_front_camera2 += 1
        

    # Count the points that are in front of both cameras
    count_in_front = np.sum(in_front_camera1 & in_front_camera2)

    return count_in_front

def create_optimization_parameters(T, triangulated_points):
    
    # Extract rotation and translation
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    t = cartesian_to_angle_vector(t)
    
    # Flatten translation and triangulated points
    t_flat = np.array(t, dtype=np.float64)
    X_w_flat = triangulated_points.ravel()
    
    # Convert rotation to so3
    rotation_vec = rot_matrix_to_so3(np.array(R, dtype=np.float64))
    
    # Create optimization parameter vector
    Op_initial = np.concatenate([rotation_vec, t_flat, X_w_flat])
    Op_initial = np.real(Op_initial)  # Ensure real values

    return Op_initial

def extract_optimized_parameters(Op, nPoints):
    # Extract optimized parameters
    optimized_R_vec = Op[:3]  # Assuming the first three elements are the rotation vector
    optimized_t = Op[3:5]      # Next three are the translation vector
    optimized_X_w = (Op[5:].reshape(nPoints,3)).T  # Remaining elements are the world points
    optimized_t = polar_to_cartesian(optimized_t)
    # Convert rotation vector to rotation matrix
    optimized_R = so3_to_rot_matrix(optimized_R_vec)

    return optimized_R, optimized_t, optimized_X_w

def polar_to_cartesian(t):
    x = np.sin(t[1]) * np.cos(t[0])
    y = np.sin(t[1]) * np.sin(t[0])
    z = np.cos(t[1])
    
    return np.array([x, y, z])

def crossMatrix(x):
    """Create an anti-symmetric matrix (cross product matrix) for a vector x."""
    M = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]], dtype="float64")
    return M

def so3_to_rot_matrix(theta):
    """Convert a rotation vector (Lie algebra so(3)) to a rotation matrix (SO(3)) using the exponential map."""
    return expm(crossMatrix(theta))

def rot_matrix_to_so3(R):
    """Convert a rotation matrix (SO(3)) to a rotation vector (Lie algebra so(3)) using the logarithmic map."""
    log_R = logm(R.astype('float64'))
    return np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])

def project_points(P, X):
    
    if X.shape[0] == 3:
        X = np.vstack((X, np.ones((1, X.shape[1]))))  

    # Project points (3x4) * (4xN) -> (3xN)
    projected = P @ X
    return projected

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    R_matrix, t_vec, X_w = extract_optimized_parameters(Op, nPoints)

    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_c @ np.hstack((R_matrix, t_vec.reshape(-1, 1)))

    projected_x1 = project_points(P1, X_w)
    projected_x2 = project_points(P2, X_w)

    residuals_x1 = (projected_x1[:2] / projected_x1[2]) - x1Data.T
    residuals_x2 = (projected_x2[:2] / projected_x2[2]) - x2Data.T

    residuals = np.hstack([residuals_x1.flatten(), residuals_x2.flatten()])

    return residuals

def cartesian_to_angle_vector(t):
    theta = math.atan2(t[1], t[0])  # Azimuthal angle
    r = math.sqrt(t[0]**2 + t[1]**2 + t[2]**2)
    
    # Avoid division by zero when calculating phi
    if r == 0:
        phi = 0  # or handle it as you prefer
    else:
        phi = math.acos(t[2] / r)  # Polar angle
    
    return [theta, phi]

def solve_pnp(object_points, image_points, K):
    """
    Solve PnP using Direct Linear Transform (DLT) followed by least squares.

    Args:
    - object_points: 3D points in world coordinates (Nx3)
    - image_points: 2D points in image coordinates (Nx2)
    - K: Camera intrinsic matrix (3x3)

    Returns:
    - rotation_matrix: 3x3 rotation matrix
    - translation_vector: 3x1 translation vector
    """
    N = object_points.shape[0]
    
    # Check if there are at least 4 points
    if N < 4:
        raise ValueError("At least 4 points are required for PnP.")
    
    # Convert to homogeneous coordinates (add a column of ones for object points)
    object_points_homogeneous = np.hstack((object_points, np.ones((N, 1))))
    image_points_homogeneous = np.hstack((image_points, np.ones((N, 1))))

    # Camera matrix K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Create the matrix A for DLT
    A = []
    for i in range(N):
        X, Y, Z = object_points[i]
        x, y = image_points[i]

        row1 = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, x]
        row2 = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, y]
        
        A.append(row1)
        A.append(row2)

    A = np.array(A)

    # Apply SVD to solve A * p = 0
    _, _, VT = svd(A)
    P = VT[-1, :]  # The last row of V^T gives us the solution for the camera matrix

    # Check if P has the correct size
    if P.shape[0] != 12:
        raise ValueError(f"Unexpected result from SVD. Expected 12 elements, but got {P.shape[0]}.")

    # Reshape into 3x4 matrix [R | t]
    P = P.reshape(3, 4)

    # Extract rotation matrix and translation vector from the camera matrix
    R = P[:, :3]
    t = P[:, 3]

    # Normalize R to get a proper rotation matrix
    # Use SVD again to ensure orthogonality of R (QR decomposition can also work)
    U, _, VT = svd(R)
    R = np.dot(U, VT)

    return R, t

def ransac_pnp(object_points, image_points, K, max_iterations=1000, threshold=1.0):
    N = object_points.shape[0]
    
    best_inliers = []
    best_R = None
    best_t = None

    for i in range(max_iterations):
        # Randomly sample 4 point correspondences
        sample_indices = np.random.choice(N, 4, replace=False)
        sample_object_points = object_points[sample_indices]
        sample_image_points = image_points[sample_indices]

        # Solve PnP using these 4 points (you can use a custom solve_pnp function here)
        R, t = solve_pnp(sample_object_points, sample_image_points, K)
        
        # Check if solve_pnp returned valid values
        if R is None or t is None:
            print("PnP solution failed for iteration", i)
            continue  # Skip this iteration if the solution is invalid

        T = ensamble_T(R, t)
        P = ensamble_P(K, T)
        projected_points = project_points(P, object_points.T)
        projected_points = projected_points[:2] / projected_points[2]
        # Compute the reprojection error for all points
        error = np.linalg.norm(projected_points - image_points.T, axis=0)

        # Find inliers (points with error less than the threshold)
        inliers = np.where(error < threshold)[0]

        # If this set of inliers is the best, update the best model
        if len(inliers) >= len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t

    if best_R is None or best_t is None:
        print("RANSAC failed to find a valid solution.")
        return None, None, []

    return best_R, best_t, best_inliers

def estimate_projection_matrix(points_3D, points_2D):
    """
    Estimate the camera projection matrix P using the Direct Linear Transformation (DLT) method.

    Args:
        points_3D: Nx3 array of 3D points in world coordinates.
        points_2D: Nx2 array of 2D points in image coordinates.

    Returns:
        P: 3x4 projection matrix.
    """
    num_points = points_3D.shape[0]
    assert points_3D.shape[0] == points_2D.shape[0], "Number of 3D and 2D points must match."

    A = []
    for i in range(num_points):
        X, Y, Z = points_3D[i]
        x, y = points_2D[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, y])

    A = np.array(A)
    
    # Solve for P using SVD
    _, _, vh = np.linalg.svd(A)
    P = vh[-1].reshape(3, 4)
    return P

def decompose_projection_matrix(P):
    """
    Decompose the projection matrix P into intrinsic matrix K, rotation matrix R_cw,
    and translation vector t_wc.

    Args:
        P: 3x4 projection matrix.

    Returns:
        K: Intrinsic matrix.
        R_cw: Rotation matrix.
        t_wc: Translation vector.
    """
    M = P[:, :3]
    t = P[:, 3]

    # Adjust M to account for sign(det(M))
    M_tilde = np.sign(np.linalg.det(M)) * M

    # Compute RQ decomposition of M_tilde
    K_hat, R_hat = np.linalg.qr(np.linalg.inv(M_tilde))
    K_hat = np.linalg.inv(K_hat)
    R_hat = np.linalg.inv(R_hat)

    # Ensure diagonals of K_hat are positive
    D = np.diag(np.sign(np.diag(K_hat)))
    R_cw = D @ R_hat
    K = np.linalg.inv(D) @ K_hat

    # Normalize K to ensure K33 = 1
    K = K / K[-1, -1]

    # Compute translation vector in the world frame
    t_wc = -R_cw @ np.linalg.inv(M_tilde) @ t

    Proj = np.sign(np.linalg.det(M))*P

    result = cv2.decomposeProjectionMatrix(Proj)
    K = result[0]
    R_cw = result[1]
    t_wc = result[2]
    K = K / K[-1, -1]

    return K, R_cw, t_wc

def computeHomography(pts1, pts2):
    """
    Compute homography using DLT algorithm from 4 points correspondences.
    """
    A = []
    for i in range(4):
        x, y = pts1[i][0], pts1[i][1]
        xp, yp = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))
    return H / H[2, 2]

def computeTransferError_Hom(H, pts1, pts2):
    """
    Compute the Euclidean distance (transfer error) between points transformed by homography and true points.
    """
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  # Convert to homogeneous coordinates
    pts1_proj = (H @ pts1_h.T).T  # Apply homography
    pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2][:, np.newaxis]  # Normalize to 2D coordinates
    errors = np.linalg.norm(pts1_proj - pts2, axis=1)  # Compute L2 distance
    return errors

def ransacHomography(pts1, pts2, n_iterations=1000, threshold=3):
    """
    RANSAC algorithm for robustly fitting a homography to two sets of points.

    Args:
    - pts1: List of points from the first image (Nx2).
    - pts2: List of corresponding points from the second image (Nx2).
    - n_iterations: Number of RANSAC iterations.
    - threshold: The threshold to classify inliers.

    Returns:
    - best_H: The best homography matrix found.
    - best_inliers: The list of inlier points that support the best homography.
    """
    best_H = None
    best_inliers = []
    n_best_inliers = 0

    for _ in range(n_iterations):
        # Randomly select 4 points
        sample_indices = random.sample(range(len(pts1)), 4)
        pts1_sample = np.array([pts1[i] for i in sample_indices])
        pts2_sample = np.array([pts2[i] for i in sample_indices])

        # Compute homography for the selected 4 points
        H = computeHomography(pts1_sample, pts2_sample)

        # Evaluate the homography on all points
        errors = computeTransferError_Hom(H, np.array(pts1), np.array(pts2))

        # Identify inliers
        inliers = [i for i in range(len(errors)) if errors[i] < threshold]

        # If this set has the most inliers, update the best model
        if len(inliers) > n_best_inliers:
            n_best_inliers = len(inliers)
            best_H = H
            best_inliers = inliers

    return best_H

def resBundleProjection_MulViews(Op, x1Data, x2Data, x3Data, x4Data, x5Data, x6Data, x7Data, x8Data, x9Data, x10Data, K_c, K_old, nPoints1, nPoints2, nPoints3, nPoints4, nPoints5, nPoints6, nPoints7, nPoints8, nPoints9):
    
    R_2, R_3, R_4, R_5, R_6, R_7, R_8, R_9, R_10, tvec_2, tvec_3,tvec_4, tvec_5, tvec_6, tvec_7, tvec_8, tvec_9, tvec_10, X_w_1, X_w_2, X_w_3, X_w_4, X_w_5, X_w_6, X_w_7, X_w_8, X_w_9 = extract_optimized_parameters_MulViews(Op, nPoints1, nPoints2, nPoints3, nPoints4, nPoints5, nPoints6, nPoints7, nPoints8, nPoints9)
    
    X_w_1 = np.vstack([X_w_1, np.ones((1, X_w_1.shape[1]))])
    X_w_2 = np.vstack([X_w_2, np.ones((1, X_w_2.shape[1]))])
    X_w_3 = np.vstack([X_w_3, np.ones((1, X_w_3.shape[1]))])
    X_w_4 = np.vstack([X_w_4, np.ones((1, X_w_4.shape[1]))])
    X_w_5 = np.vstack([X_w_5, np.ones((1, X_w_5.shape[1]))])
    X_w_6 = np.vstack([X_w_6, np.ones((1, X_w_6.shape[1]))])
    X_w_7 = np.vstack([X_w_7, np.ones((1, X_w_7.shape[1]))])
    X_w_8 = np.vstack([X_w_8, np.ones((1, X_w_8.shape[1]))])
    X_w_9 = np.vstack([X_w_9, np.ones((1, X_w_9.shape[1]))])
    
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_old @ np.hstack((R_2, tvec_2.reshape(-1, 1)))
    P3 = K_c @ np.hstack((R_3, tvec_3.reshape(-1, 1)))
    P4 = K_c @ np.hstack((R_4, tvec_4.reshape(-1, 1)))
    P5 = K_c @ np.hstack((R_5, tvec_5.reshape(-1, 1)))
    P6 = K_c @ np.hstack((R_6, tvec_6.reshape(-1, 1)))
    P7 = K_c @ np.hstack((R_7, tvec_7.reshape(-1, 1)))
    P8 = K_c @ np.hstack((R_8, tvec_8.reshape(-1, 1)))
    P9 = K_c @ np.hstack((R_9, tvec_9.reshape(-1, 1)))
    P10 = K_c @ np.hstack((R_10, tvec_10.reshape(-1, 1)))

    projected_x1 = project_points(P1, X_w_1)
    projected_x2 = project_points(P2, X_w_1)
    projected_x3 = project_points(P3, X_w_2)
    projected_x4 = project_points(P4, X_w_3)
    projected_x5 = project_points(P5, X_w_4)
    projected_x6 = project_points(P6, X_w_5)
    projected_x7 = project_points(P7, X_w_6) 
    projected_x8 = project_points(P8, X_w_7)
    projected_x9 = project_points(P9, X_w_8)
    projected_x10 = project_points(P10, X_w_9) 

    x10Data = x10Data[:-1]

    residuals_x1 = (projected_x1[:2] / projected_x1[2]) - x1Data.T[:2]
    residuals_x2 = (projected_x2[:2] / projected_x2[2]) - x2Data.T[:2]
    residuals_x3 = (projected_x3[:2] / projected_x3[2]) - x3Data.T[:2]
    residuals_x4 = (projected_x4[:2] / projected_x4[2]) - x4Data.T[:2]
    residuals_x5 = (projected_x5[:2] / projected_x5[2]) - x5Data.T[:2]
    residuals_x6 = (projected_x6[:2] / projected_x6[2]) - x6Data.T[:2] 
    residuals_x7 = (projected_x7[:2] / projected_x7[2]) - x7Data.T[:2]
    residuals_x8 = (projected_x8[:2] / projected_x8[2]) - x8Data.T[:2]
    residuals_x9 = (projected_x9[:2] / projected_x9[2]) - x9Data.T[:2] 
    residuals_x10 = (projected_x10[:2] / projected_x10[2]) - x10Data.T[:2]
       

    residuals = np.hstack([residuals_x1.flatten(), residuals_x2.flatten(), residuals_x3.flatten(),residuals_x4.flatten(), residuals_x5.flatten(), residuals_x6.flatten(),residuals_x7.flatten(), residuals_x8.flatten(), residuals_x9.flatten(), residuals_x10.flatten()])

    return residuals

def create_optimization_parameters_MulViews(T1, T2, T3, T4, T5, T6, T7, T8, T9, triangulated_points_1, triangulated_points_2, triangulated_points_3, triangulated_points_4, triangulated_points_5, triangulated_points_6, triangulated_points_7, triangulated_points_8, triangulated_points_9 ):
    # Extract rotation and translation
    R_1 = T1[0:3, 0:3]
    t_1 = T1[0:3, 3]
    t_1 = cartesian_to_angle_vector(t_1)
    
    R_2 = T2[0:3, 0:3]
    t_2 = T2[0:3, 3]

    R_3 = T3[0:3, 0:3]
    t_3 = T3[0:3, 3]

    R_4 = T4[0:3, 0:3]
    t_4 = T4[0:3, 3]

    R_5 = T5[0:3, 0:3]
    t_5 = T5[0:3, 3]

    R_6 = T6[0:3, 0:3]
    t_6 = T6[0:3, 3]

    R_7 = T7[0:3, 0:3]
    t_7 = T7[0:3, 3]

    R_8 = T8[0:3, 0:3]
    t_8 = T8[0:3, 3]

    R_9 = T9[0:3, 0:3]
    t_9 = T9[0:3, 3]

    # Flatten translation and triangulated points
    t_flat_1 = np.array(t_1, dtype=np.float64)
    t_flat_2 = np.array(t_2, dtype=np.float64)
    t_flat_3 = np.array(t_3, dtype=np.float64)
    t_flat_4 = np.array(t_4, dtype=np.float64)
    t_flat_5 = np.array(t_5, dtype=np.float64)
    t_flat_6 = np.array(t_6, dtype=np.float64)
    t_flat_7 = np.array(t_7, dtype=np.float64)
    t_flat_8 = np.array(t_8, dtype=np.float64)
    t_flat_9 = np.array(t_9, dtype=np.float64)
    X_flat_1 = triangulated_points_1.ravel()
    X_flat_2 = triangulated_points_2.ravel()
    X_flat_3 = triangulated_points_3.ravel()
    X_flat_4 = triangulated_points_4.ravel()
    X_flat_5 = triangulated_points_5.ravel()
    X_flat_6 = triangulated_points_6.ravel()
    X_flat_7 = triangulated_points_7.ravel()
    X_flat_8 = triangulated_points_8.ravel()
    X_flat_9 = triangulated_points_9.ravel()  

    # Convert rotation to so3
    rotation_vec_1 = rot_matrix_to_so3(np.array(R_1, dtype=np.float64))
    rotation_vec_2 = rot_matrix_to_so3(np.array(R_2, dtype=np.float64))
    rotation_vec_3 = rot_matrix_to_so3(np.array(R_3, dtype=np.float64))
    rotation_vec_4 = rot_matrix_to_so3(np.array(R_4, dtype=np.float64)) 
    rotation_vec_5 = rot_matrix_to_so3(np.array(R_5, dtype=np.float64))
    rotation_vec_6 = rot_matrix_to_so3(np.array(R_6, dtype=np.float64)) 
    rotation_vec_7 = rot_matrix_to_so3(np.array(R_7, dtype=np.float64))
    rotation_vec_8 = rot_matrix_to_so3(np.array(R_8, dtype=np.float64)) 
    rotation_vec_9 = rot_matrix_to_so3(np.array(R_9, dtype=np.float64))
     

    # Create optimization parameter vector
    Op_initial = np.concatenate([rotation_vec_1, rotation_vec_2, rotation_vec_3, rotation_vec_4, rotation_vec_5, rotation_vec_6, rotation_vec_7, rotation_vec_8, rotation_vec_9, t_flat_1, t_flat_2, t_flat_3, t_flat_4, t_flat_5, t_flat_6, t_flat_7, t_flat_8, t_flat_9, X_flat_1, X_flat_2 , X_flat_3, X_flat_4 , X_flat_5, X_flat_6 , X_flat_7, X_flat_8 , X_flat_9])
    Op_initial = np.real(Op_initial)  # Ensure real values

    return Op_initial


def extract_optimized_parameters_MulViews(Op, nPoints1, nPoints2, nPoints3, nPoints4, nPoints5, nPoints6, nPoints7, nPoints8, nPoints9):
    # Extract optimized parameters
    optimized_R_vec_1 = Op[:3]  
    optimized_R_vec_2 = Op[3:6]
    optimized_R_vec_3 = Op[6:9]  
    optimized_R_vec_4 = Op[9:12]  
    optimized_R_vec_5 = Op[12:15]
    optimized_R_vec_6 = Op[18:21]  
    optimized_R_vec_7 = Op[21:24]
    optimized_R_vec_8 = Op[24:27]  
    optimized_R_vec_9 = Op[27:30]  
    optimized_t_1 = Op[30:32]      
    optimized_t_2 = Op[32:35]
    optimized_t_3 = Op[35:38]      
    optimized_t_4 = Op[38:41]
    optimized_t_5 = Op[41:44]      
    optimized_t_6 = Op[44:47]
    optimized_t_7 = Op[47:50]      
    optimized_t_8 = Op[50:53]
    optimized_t_9 = Op[53:56]           
    optimized_X_w_1 = Op[56:56 + 3 * nPoints1].reshape(nPoints1, 3).T  
    optimized_X_w_2 = Op[56 + 3 * nPoints1:56 + 3 * nPoints1 + 3*nPoints2 ].reshape(nPoints2, 3).T 
    optimized_X_w_3 = Op[56 + 3 * nPoints1 + 3*nPoints2:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3 ].reshape(nPoints3, 3).T   
    optimized_X_w_4 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 ].reshape(nPoints4, 3).T
    optimized_X_w_5 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 :56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5].reshape(nPoints5, 3).T
    optimized_X_w_6 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6].reshape(nPoints6, 3).T   
    optimized_X_w_7 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6 + 3*nPoints7].reshape(nPoints7, 3).T
    optimized_X_w_8 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6 + 3*nPoints7:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6 + 3*nPoints7+3*nPoints8].reshape(nPoints8, 3).T 
    optimized_X_w_9 = Op[56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6 + 3*nPoints7+3*nPoints8:56 + 3 * nPoints1 + 3*nPoints2+3*nPoints3+3*nPoints4 + 3*nPoints5 +3*nPoints6 + 3*nPoints7+3*nPoints8+3*nPoints9].reshape((nPoints9-1), 3).T  
    
    optimized_t_1 = polar_to_cartesian(optimized_t_1)
    
    optimized_R_1 = so3_to_rot_matrix(optimized_R_vec_1)
    optimized_R_2 = so3_to_rot_matrix(optimized_R_vec_2)
    optimized_R_3 = so3_to_rot_matrix(optimized_R_vec_3)
    optimized_R_4 = so3_to_rot_matrix(optimized_R_vec_4)
    optimized_R_5 = so3_to_rot_matrix(optimized_R_vec_5)
    optimized_R_6 = so3_to_rot_matrix(optimized_R_vec_6)
    optimized_R_7 = so3_to_rot_matrix(optimized_R_vec_7)
    optimized_R_8 = so3_to_rot_matrix(optimized_R_vec_8)
    optimized_R_9 = so3_to_rot_matrix(optimized_R_vec_9)
    

    return optimized_R_1, optimized_R_2, optimized_R_3, optimized_R_4, optimized_R_5, optimized_R_6, optimized_R_7, optimized_R_8, optimized_R_9, optimized_t_1,optimized_t_2,optimized_t_3,optimized_t_4, optimized_t_5,optimized_t_6, optimized_t_7,optimized_t_8, optimized_t_9, optimized_X_w_1, optimized_X_w_2, optimized_X_w_3, optimized_X_w_4 , optimized_X_w_5, optimized_X_w_6 , optimized_X_w_7, optimized_X_w_8, optimized_X_w_9

#-------------------------------------------------------------------------------------------------------------------

#DEBUG FUNCTION

def getIndex3D(point, listPoints):
    for i in range(len(listPoints.T)):
        if np.all(listPoints.T[i] == point):
            return i
    else:
        return -1

def getIndex2D(point, listPoints):
    for i in range(len(listPoints)):
        if np.all(listPoints[i] == point):
            return i
    else:
        return -1

#-------------------------------------------------------------------------------------------------------------------------------------

#DIFFERENCES IN IMAGES

#-------------------------------------------------------------------------------------------------------------------------------------

def resize_maintain_aspect_ratio(image, target_height, target_width):
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    # Resize based on the target dimensions while maintaining aspect ratio
    if target_width / target_height > aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image

def image_difference(image1, image2, threshold=30):
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a threshold to highlight significant differences
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Optionally, you can use contour detection to find the areas of difference
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around the differences
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust based on your scenario
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return image1, thresh

def calculate_ssim(image1, image2):
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between two images
    score, diff = ssim(gray1, gray2, full=True)
    
    # Normalize the diff image to [0, 255] range
    diff = (diff * 255).astype("uint8")
    
    return score, diff

def background_subtraction_simple(image1, image2, threshold=30):
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference image to highlight significant changes
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    return thresh

def edge_detection(image, low_threshold=100, high_threshold=200):
    """
    Perform edge detection using the Canny algorithm.

    Args:
        image (numpy.ndarray): The input image on which to apply edge detection.
        low_threshold (int): The lower bound for the Canny edge detector.
        high_threshold (int): The upper bound for the Canny edge detector.

    Returns:
        numpy.ndarray: The edge-detected image.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges

def detect_difference(image1, image2):
    """
    Detect differences between two aligned images by subtracting them.
    
    Args:
        image1 (numpy.ndarray): The first image (reference).
        image2 (numpy.ndarray): The second image (aligned version).
        
    Returns:
        numpy.ndarray: The difference image.
    """
    # Step 1: Compute the absolute difference
    diff = cv2.absdiff(image1, image2)
    
    # Step 2: Convert the difference to grayscale (if necessary)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Threshold the difference to highlight significant changes
    _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    
    return thresh_diff

def colorize_edges(edges, color):
    """
    Colorize the edge-detected image with a specified color.
    
    Args:
        edges (numpy.ndarray): The edge-detected binary image.
        color (tuple): The color to apply to the edges (B, G, R).
        
    Returns:
        numpy.ndarray: The colorized edge image.
    """
    # Create an empty color image (3 channels)
    color_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

    # Assign the color to the edge positions (where edges are non-zero)
    color_image[edges != 0] = color

    return color_image

#-------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    K = np.array([[821.78396784  , 0.    ,     445.91316442],
                [  0.     ,    833.61565894, 459.80590661],
                [  0.     ,      0.       ,    1.        ]])

    print('Camera calibration: \n', K)

    dist_coeffs = np.zeros(4, dtype=np.float32)            

    Image = cv2.imread('Undistorted_Images/undistorted_CVFotoActual.jpeg')
    image0 = cv2.imread('Undistorted_Images/CVFoto0.jpeg')
    image1 = cv2.imread('Undistorted_Images/undistorted_CVFoto1.jpeg')
    image2 = cv2.imread('Undistorted_Images/undistorted_CVFoto2.jpeg')
    image3 = cv2.imread('Undistorted_Images/undistorted_CVFoto3.jpeg')
    image4 = cv2.imread('Undistorted_Images/undistorted_CVFoto4.jpeg')
    image5 = cv2.imread('Undistorted_Images/undistorted_CVFoto5.jpeg')
    image6 = cv2.imread('Undistorted_Images/undistorted_CVFoto6.jpeg')
    image7 = cv2.imread('Undistorted_Images/undistorted_CVFoto7.jpeg')
    image8 = cv2.imread('Undistorted_Images/undistorted_CVFoto8.jpeg')
    image9 = cv2.imread('Undistorted_Images/undistorted_CVFoto9.jpeg')
    image10 = cv2.imread('Undistorted_Images/undistorted_CVFoto10.jpeg')
    image11 = cv2.imread('Undistorted_Images/undistorted_CVFoto11.jpeg')
    image12 = cv2.imread('Undistorted_Images/undistorted_CVFoto12.jpeg')
    image13 = cv2.imread('Undistorted_Images/undistorted_CVFoto13.jpeg')
    image14 = cv2.imread('Undistorted_Images/undistorted_CVFoto14.jpeg')
    image15 = cv2.imread('Undistorted_Images/undistorted_CVFoto15.jpeg')

    kp1_0, kp2_0, matches_0, desc1_0, desc2_0 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_CVFoto0_matches.npz')
    kp1_1, kp2_1, matches_1, desc1_1, desc2_1 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto1_matches.npz')
    kp1_2, kp2_2, matches_2, desc1_2, desc2_2 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto2_matches.npz')
    kp1_3, kp2_3, matches_3, desc1_3, desc2_3 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto3_matches.npz')
    kp1_4, kp2_4, matches_4, desc1_4, desc2_4 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto4_matches.npz')
    kp1_5, kp2_5, matches_5, desc1_5, desc2_5 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto5_matches.npz')
    kp1_6, kp2_6, matches_6, desc1_6, desc2_6 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto6_matches.npz')
    kp1_7, kp2_7, matches_7, desc1_7, desc2_7 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto7_matches.npz')
    kp1_8, kp2_8, matches_8, desc1_8, desc2_8 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto8_matches.npz')
    kp1_9, kp2_9, matches_9, desc1_9, desc2_9 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto9_matches.npz')
    kp1_10, kp2_10, matches_10, desc1_10, desc2_10 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto10_matches.npz')
    kp1_11, kp2_11, matches_11, desc1_11, desc2_11 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto11_matches.npz')
    kp1_12, kp2_12, matches_12, desc1_12, desc2_12 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto12_matches.npz')
    kp1_13, kp2_13, matches_13, desc1_13, desc2_13 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto13_matches.npz')
    kp1_14, kp2_14, matches_14, desc1_14, desc2_14 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto14_matches.npz')
    kp1_15, kp2_15, matches_15, desc1_15, desc2_15 = extract_keypoints_descriptors('./SGmatches_undistorted/undistorted_CVFotoActual_undistorted_CVFoto15_matches.npz')

    kp1_0_match, kp2_0_match = extract_matches(kp1_0, kp2_0, matches_0)

    original_shape = (1257, 828)  # Original image dimensions (height, width)
    distorted_shape = (750, 1000)
    kp2_0_match = warp_points_back_to_original(kp2_0_match, original_shape, distorted_shape)

    kp1_1_match, kp2_1_match = extract_matches(kp1_1, kp2_1, matches_1)
    kp1_2_match, kp2_2_match = extract_matches(kp1_2, kp2_2, matches_2)
    kp1_3_match, kp2_3_match = extract_matches(kp1_3, kp2_3, matches_3)
    kp1_4_match, kp2_4_match = extract_matches(kp1_4, kp2_4, matches_4)
    kp1_5_match, kp2_5_match = extract_matches(kp1_5, kp2_5, matches_5)
    kp1_6_match, kp2_6_match = extract_matches(kp1_6, kp2_6, matches_6)
    kp1_7_match, kp2_7_match = extract_matches(kp1_7, kp2_7, matches_7)
    kp1_8_match, kp2_8_match = extract_matches(kp1_8, kp2_8, matches_8)
    kp1_9_match, kp2_9_match = extract_matches(kp1_9, kp2_9, matches_9)
    kp1_10_match, kp2_10_match = extract_matches(kp1_10, kp2_10, matches_10)
    kp1_11_match, kp2_11_match = extract_matches(kp1_11, kp2_11, matches_11)
    kp1_12_match, kp2_12_match = extract_matches(kp1_12, kp2_12, matches_12)
    kp1_13_match, kp2_13_match = extract_matches(kp1_13, kp2_13, matches_13)
    kp1_14_match, kp2_14_match = extract_matches(kp1_14, kp2_14, matches_14)
    kp1_15_match, kp2_15_match = extract_matches(kp1_15, kp2_15, matches_15)

    common_kp1_6_15 = get_common_keypoints([kp1_6_match, kp1_15_match])
    common_kp2_15= get_common_kp2(kp1_15_match, kp2_15_match, common_kp1_6_15)

    common_kp1_6_1 = get_common_keypoints([kp1_6_match, kp1_1_match])
    common_kp2_1= get_common_kp2(kp1_1_match, kp2_1_match, common_kp1_6_1)

    common_kp1_old_6 = get_common_keypoints([kp1_0_match, kp1_6_match])
    common_kp2_6= get_common_kp2(kp1_6_match, kp2_6_match, common_kp1_old_6)
    common_kp2_0 = get_common_kp2(kp1_0_match, kp2_0_match, common_kp1_old_6)
    
    common_kp1_6_12 = get_common_keypoints([kp1_6_match, kp1_12_match])
    common_kp2_12= get_common_kp2(kp1_12_match, kp2_12_match, common_kp1_6_12)

    common_kp1_6_3 = get_common_keypoints([kp1_6_match, kp1_3_match])
    common_kp2_3 = get_common_kp2(kp1_3_match, kp2_3_match, common_kp1_6_3)

    common_kp1_6_8 = get_common_keypoints([kp1_6_match, kp1_8_match])
    common_kp2_8 = get_common_kp2(kp1_8_match, kp2_8_match, common_kp1_6_8)

    common_kp1_6_9 = get_common_keypoints([kp1_6_match, kp1_9_match])
    common_kp2_9 = get_common_kp2(kp1_9_match, kp2_9_match, common_kp1_6_9)

    common_kp1_6_14 = get_common_keypoints([kp1_6_match, kp1_14_match])
    common_kp2_14 = get_common_kp2(kp1_14_match, kp2_14_match, common_kp1_6_14)

    common_kp1_all = get_common_keypoints([kp1_0_match, kp1_6_match, kp1_14_match, kp1_15_match, kp1_1_match, kp1_12_match, kp1_8_match, kp1_9_match, kp1_3_match])
    common_kp2_all = get_common_kp2(kp1_5_match, kp2_5_match, common_kp1_all)



    #-------------------------------------------------------------------------------------------------------------------------------

     #Display both images and set up the click event to plot epipolar lines
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display Image ref with points
    ax1.imshow(Image, cmap='gray')
    ax1.plot(np.array(kp1_6_match)[:, 0], np.array(kp1_6_match)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(kp1_6_match), 'r', (10, 0))  # Plot numbered points
    ax1.set_title('Image ref')

    # Display Image 6 with points
    ax2.imshow(image6, cmap='gray')
    ax2.plot(np.array(kp2_6_match)[:, 0], np.array(kp2_6_match)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(kp2_6_match), 'r', (10, 0))  # Plot numbered points
    ax2.set_title('Image 6')

    #----------------------------------------------------------------------------------------------------------------

    F_ref_6 = ransac_fundamental_matrix(kp1_6_match, kp2_6_match, 1000, 1)

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, Image, image6, F_ref_6, ax1, ax2))

    plt.show()

    #---------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display Image ref with points
    ax1.imshow(Image, cmap='gray')
    ax1.plot(np.array(kp1_0_match)[:, 0], np.array(kp1_0_match)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(kp1_0_match), 'r', (10, 0))  # Plot numbered points
    ax1.set_title('Image ref')

    # Display Image 6 with points
    ax2.imshow(image0, cmap='gray')
    ax2.plot(np.array(kp2_0_match)[:, 0], np.array(kp2_0_match)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(kp2_0_match), 'r', (10, 0))  # Plot numbered points
    ax2.set_title('Image 0')

    F_ref_0 = ransac_fundamental_matrix(kp1_0_match, kp2_0_match, 1000, 1)

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, Image, image0, F_ref_0, ax1, ax2))

    plt.show()

    #--------------------------------------------------------------------------------------------------------------------------

    x1 = np.array(kp1_6_match)
    x2 = np.array(kp2_6_match)

    E_ref_6 = fundamental_To_Essential(F_ref_6, K)
    poses = decompose_essential_matrix(E_ref_6)

    T_candidates = []
    triangulated_points = []
    
    Matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    
    P_ref = K @ Matrix

    for i, (R, t) in enumerate(poses): 
        T = ensamble_T(R, t)
        T_candidates.append(T)
        print("Candidate Transformation T:\n", T)

        P2 = ensamble_P(K, T)
        points_3d = triangulate_points(P_ref, P2, x1, x2)
        triangulated_points.append(points_3d)

    best_T, best_points = best_pose_estimation(T_candidates, triangulated_points, P_ref, K)
    print("Best Transformation Matrix:\n", best_T)

    T_6 = best_T

    #------------------------------------------------------------------------------------------------------

    Op_initial = create_optimization_parameters(T_6, best_points)
    nPoints = x1.shape[0]

    Op_initial = np.real(Op_initial)
    
    # Run least squares optimization
    result = least_squares(resBundleProjection, Op_initial, args=(x1, x2, K, nPoints))

    # Extract optimized parameters
    optimized_R, optimized_t, optimized_X = extract_optimized_parameters(result.x, nPoints)
    
    print("Optimized Rotation Matrix:\n", optimized_R)
    print("Optimized Translation Vector:\n", optimized_t)
    
    # Plot the 3D points after bundle adjustment
    fig3D = plt.figure(1)
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw original world reference system
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')

    # Plot optimized camera positions
    # Note: Since `optimized_R` and `optimized_t` correspond to the optimized camera pose, assemble the transformation matrix for visualization
    T_6 = np.eye(4)
    T_6[:3, :3] = optimized_R
    T_6[:3, 3] = -optimized_t
    drawRefSystem(ax, T_6, '-', 'C_6')

    # Plot optimized 3D points
    ax.scatter(optimized_X[0, :], optimized_X[1, :], optimized_X[2, :], marker='o', color='b')
    #plotNumbered3DPoints(ax, optimized_X_w, 'r', 0.1)  # Plot optimized points in red

    # Matplotlib bounding box adjustment
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    print('Close the figure to continue.')
    plt.show()

    #--------------------------------------------------------------------------------------------------

    threeDPoints_15 = []

    for i in range(len(common_kp1_6_15)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_15[i] == kp1_6_match[a]):
                threeDPoints_15.append(optimized_X.T[a])

    threeDPoints_15 = np.array(threeDPoints_15)

    # idx_3 = getIndex3D(threeDPoints_15[0], optimized_X)
    # idx_2 = getIndex2D(kp1_6_match[idx_3], common_kp1_6_15)

    threeDPoints_1 = []

    for i in range(len(common_kp1_6_1)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_1[i] == kp1_6_match[a]):
                threeDPoints_1.append(optimized_X.T[a])

    threeDPoints_1 = np.array(threeDPoints_1)

    threeDPoints_12 = []

    for i in range(len(common_kp1_6_12)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_12[i] == kp1_6_match[a]):
                threeDPoints_12.append(optimized_X.T[a])

    threeDPoints_12 = np.array(threeDPoints_12)

    threeDPoints_3 = []

    for i in range(len(common_kp1_6_3)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_3[i] == kp1_6_match[a]):
                threeDPoints_3.append(optimized_X.T[a])

    threeDPoints_3 = np.array(threeDPoints_3)

    threeDPoints_8 = []

    for i in range(len(common_kp1_6_8)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_8[i] == kp1_6_match[a]):
                threeDPoints_8.append(optimized_X.T[a])

    threeDPoints_8 = np.array(threeDPoints_8)

    threeDPoints_9 = []

    for i in range(len(common_kp1_6_9)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_9[i] == kp1_6_match[a]):
                threeDPoints_9.append(optimized_X.T[a])

    threeDPoints_9 = np.array(threeDPoints_9)

    threeDPoints_14 = []

    for i in range(len(common_kp1_6_14)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_6_14[i] == kp1_6_match[a]):
                threeDPoints_14.append(optimized_X.T[a])

    threeDPoints_14 = np.array(threeDPoints_14)

    #-------------------------------------------------------------------------------------------------------

    R_15, t_15, inliers = ransac_pnp(threeDPoints_15, np.array(common_kp2_15), K, max_iterations=1000, threshold=1.0)

    # Use PnP with RANSAC
    success, R_15, t_15, inliers = cv2.solvePnPRansac(
    threeDPoints_15, np.array(common_kp2_15), K, dist_coeffs)

    R_15, _ = cv2.Rodrigues(R_15)

    T_15 = np.eye(4)
    T_15[:3, :3] = R_15
    T_15[:3, 3] = -t_15.flatten()

    print(T_15)

    R_1, t_1, inliers = ransac_pnp(threeDPoints_1, np.array(common_kp2_1), K, max_iterations=1000, threshold=1.0)

    # Use PnP with RANSAC
    success, R_1, t_1, inliers = cv2.solvePnPRansac(
    threeDPoints_1, np.array(common_kp2_1), K, dist_coeffs)

    R_1, _ = cv2.Rodrigues(R_1)

    T_1 = np.eye(4)
    T_1[:3, :3] = R_1
    T_1[:3, 3] = -t_1.flatten()

    print(T_1)

    R_12, t_12, inliers = ransac_pnp(threeDPoints_12, np.array(common_kp2_12), K, max_iterations=1000, threshold=1.0)

    success, R_12, t_12, inliers = cv2.solvePnPRansac(
    threeDPoints_12, np.array(common_kp2_12), K, dist_coeffs)

    R_12, _ = cv2.Rodrigues(R_12)

    T_12 = np.eye(4)
    T_12[:3, :3] = R_12
    T_12[:3, 3] = -t_12.flatten()

    print(T_12)

    R_3, t_3, inliers = ransac_pnp(threeDPoints_3, np.array(common_kp2_3), K, max_iterations=1000, threshold=1.0)

    success, R_3, t_3, inliers = cv2.solvePnPRansac(
    threeDPoints_3, np.array(common_kp2_3), K, dist_coeffs)

    R_3, _ = cv2.Rodrigues(R_3)

    T_3 = np.eye(4)
    T_3[:3, :3] = R_3
    T_3[:3, 3] = -t_3.flatten()

    print(T_3)

    R_8, t_8, inliers = ransac_pnp(threeDPoints_8, np.array(common_kp2_8), K, max_iterations=1000, threshold=1.0)

    success, R_8, t_8, inliers = cv2.solvePnPRansac(
    threeDPoints_8, np.array(common_kp2_8), K, dist_coeffs)

    R_8, _ = cv2.Rodrigues(R_8)

    T_8 = np.eye(4)
    T_8[:3, :3] = R_8
    T_8[:3, 3] = -t_8.flatten()

    print(T_8)

    R_9, t_9, inliers = ransac_pnp(threeDPoints_9, np.array(common_kp2_9), K, max_iterations=1000, threshold=1.0)

    success, R_9, t_9, inliers = cv2.solvePnPRansac(
    threeDPoints_9, np.array(common_kp2_9), K, dist_coeffs)

    R_9, _ = cv2.Rodrigues(R_9)

    T_9 = np.eye(4)
    T_9[:3, :3] = R_9
    T_9[:3, 3] = -t_9.flatten()

    print(T_9)

    R_14, t_14, inliers = ransac_pnp(threeDPoints_14, np.array(common_kp2_14), K, max_iterations=1000, threshold=1.0)

    success, R_14, t_14, inliers = cv2.solvePnPRansac(
    threeDPoints_14, np.array(common_kp2_14), K, dist_coeffs)

    R_14, _ = cv2.Rodrigues(R_14)

    T_14 = np.eye(4)
    T_14[:3, :3] = R_14
    T_14[:3, 3] = -t_14.flatten()

    print(T_14)

    #--------------------------------------------------------------------------------------------------------------

    fig3D = plt.figure(1)
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw original world reference system
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_6, '-', 'C_6')
    drawRefSystem(ax, T_15, '-', 'C_15')
    drawRefSystem(ax, T_1, '-', 'C_1')
    drawRefSystem(ax, T_12, '-', 'C_12')
    drawRefSystem(ax, T_3, '-', 'C_3')
    drawRefSystem(ax, T_8, '-', 'C_8')
    drawRefSystem(ax, T_9, '-', 'C_9')
    drawRefSystem(ax, T_14, '-', 'C_14')

    ax.scatter(optimized_X[0, :], optimized_X[1, :], optimized_X[2, :], marker='o', color='b')

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    print('Close the figure to continue.')
    plt.show()

    #-------------------------------------------------------------------------------------------------------------------------

    threeDPoints_0 = []

    for i in range(len(common_kp1_old_6)):
        for a in range(len(kp1_6_match)):
            if np.all(common_kp1_old_6[i] == kp1_6_match[a]):
                threeDPoints_0.append(optimized_X.T[a])

    threeDPoints_0 = np.array(threeDPoints_0)

    #-------------------------------------------------------------------------------------------------------------------------

    P = estimate_projection_matrix(threeDPoints_0, np.array(common_kp2_0))
    K_old, R_old, t_old = decompose_projection_matrix(P)

    K_old[0,1] = 0

    T_old = np.eye(4)
    T_old[:3, :3] = R_old
    T_old[:3, 3] = t_old[:3].flatten()

    print("Old translation: \n", T_old)
    print("Old calibration: \n", K_old)

    fig3D = plt.figure(1)
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw original world reference system
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_old, '-', 'C_old')
    # drawRefSystem(ax, T_6, '-', 'C_6')
    # drawRefSystem(ax, T_15, '-', 'C_15')
    # drawRefSystem(ax, T_1, '-', 'C_1')
    # drawRefSystem(ax, T_12, '-', 'C_12')
    # drawRefSystem(ax, T_3, '-', 'C_3')
    # drawRefSystem(ax, T_8, '-', 'C_8')
    # drawRefSystem(ax, T_9, '-', 'C_9')
    # drawRefSystem(ax, T_14, '-', 'C_14')

    ax.scatter(optimized_X[0, :], optimized_X[1, :], optimized_X[2, :], marker='o', color='b')

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    print('Close the figure to continue.')
    plt.show()

    #--------------------------------------------------------------------------------------------------------------------------------

    #BUNDLE ADJUSTMENT MULTIVIEW

    #--------------------------------------------------------------------------------------------------------------------------------

    P_old = ensamble_P(K_old, T_old)
    P_6 = ensamble_P(K, T_6)
    P_15 = ensamble_P(K, T_15)
    P_1 = ensamble_P(K, T_1)
    P_12 = ensamble_P(K, T_12)
    P_3 = ensamble_P(K, T_3)
    P_8 = ensamble_P(K, T_8)
    P_9 = ensamble_P(K, T_9)
    P_14 = ensamble_P(K, T_14)

    trian_points_old = triangulate_points(P_ref,P_old, np.array(kp1_0_match),np.array(kp2_0_match))
    trian_points_6 = triangulate_points(P_ref,P_6, np.array(kp1_6_match),np.array(kp2_6_match))
    trian_points_15 = triangulate_points(P_ref,P_15, np.array(kp1_15_match),np.array(kp2_15_match))
    trian_points_1 = triangulate_points(P_ref,P_1,np.array(kp1_1_match),np.array(kp2_1_match))
    trian_points_12 = triangulate_points(P_ref,P_12, np.array(kp1_12_match),np.array(kp2_12_match))
    trian_points_3 = triangulate_points(P_ref,P_3, np.array(kp1_3_match),np.array(kp2_3_match))
    trian_points_8 = triangulate_points(P_ref,P_8, np.array(kp1_8_match),np.array(kp2_8_match))
    trian_points_9 = triangulate_points(P_ref,P_9, np.array(kp1_9_match),np.array(kp2_9_match))
    trian_points_14 = triangulate_points(P_ref,P_14, np.array(kp1_14_match),np.array(kp2_14_match))

    Op_initial = create_optimization_parameters_MulViews(T_old, T_6, T_15, T_1, T_12, T_3, T_8, T_9, T_14, trian_points_old,trian_points_6,trian_points_15,trian_points_1,trian_points_12,trian_points_3,trian_points_8,trian_points_9,trian_points_14)
    nPoints1 = len(kp2_0_match)
    nPoints2 = len(kp2_6_match)
    nPoints3 = len(kp2_15_match)
    nPoints4 = len(kp2_1_match)
    nPoints5 = len(kp2_12_match)
    nPoints6 = len(kp2_3_match)
    nPoints7 = len(kp2_8_match)
    nPoints8 = len(kp2_9_match)
    nPoints9 = len(kp2_14_match)


    Op_initial = np.real(Op_initial)
    
    # Run least squares optimization
    result = least_squares(resBundleProjection_MulViews, Op_initial, args=(np.array(kp1_0_match), np.array(kp2_0_match), np.array(kp2_6_match), np.array(kp2_15_match), np.array(kp2_1_match), np.array(kp2_12_match), np.array(kp2_3_match), np.array(kp2_8_match), np.array(kp2_9_match), np.array(kp2_14_match), K,K_old, nPoints1, nPoints2, nPoints3, nPoints4, nPoints5, nPoints6, nPoints7, nPoints8, nPoints9))

    # Extract optimized parameters
    optimized_R_0, optimized_R_6, optimized_R_15, optimized_R_1, optimized_R_12, optimized_R_3, optimized_R_8, optimized_R_9, optimized_R_14, optimized_t_0,optimized_t_6,optimized_t_15,optimized_t_1, optimized_t_12,optimized_t_3, optimized_t_8,optimized_t_9, optimized_t_14, optimized_X_w_0, optimized_X_w_6, optimized_X_w_15, optimized_X_w_1 , optimized_X_w_12, optimized_X_w_3 , optimized_X_w_8, optimized_X_w_9, optimized_X_w_14 = extract_optimized_parameters_MulViews(result.x, nPoints1, nPoints2, nPoints3, nPoints4, nPoints5, nPoints6, nPoints7, nPoints8, nPoints9)

    T_old[:3, :3] = optimized_R_0
    T_old[:3, 3] = -optimized_t_0

    T_6[:3, :3] = optimized_R_6
    T_6[:3, 3] = -optimized_t_6

    T_15[:3, :3] = optimized_R_15
    T_15[:3, 3] = -optimized_t_15

    T_1[:3, :3] = optimized_R_1
    T_1[:3, 3] = -optimized_t_1

    T_12[:3, :3] = optimized_R_12
    T_12[:3, 3] = -optimized_t_12

    T_3[:3, :3] = optimized_R_3
    T_3[:3, 3] = -optimized_t_3

    T_8[:3, :3] = optimized_R_8
    T_8[:3, 3] = -optimized_t_8

    T_9[:3, :3] = optimized_R_9
    T_9[:3, 3] = -optimized_t_9

    T_14[:3, :3] = optimized_R_14
    T_14[:3, 3] = -optimized_t_14

    fig3D = plt.figure(1)
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw original world reference system
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_old, '-', 'C_old')
    drawRefSystem(ax, T_6, '-', 'C_6')
    drawRefSystem(ax, T_15, '-', 'C_15')
    drawRefSystem(ax, T_1, '-', 'C_1')
    drawRefSystem(ax, T_12, '-', 'C_12')
    drawRefSystem(ax, T_3, '-', 'C_3')
    drawRefSystem(ax, T_8, '-', 'C_8')
    drawRefSystem(ax, T_9, '-', 'C_9')
    drawRefSystem(ax, T_14, '-', 'C_14')

    ax.scatter(optimized_X_w_6[0, :], optimized_X_w_6[1, :], optimized_X_w_6[2, :], marker='o', color='b')

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    print('Close the figure to continue.')
    plt.show()
    #----------------------------------------------------------------------------------------------------------------------------

    #FIND DIFFERENCES BETWEEN IMAGES

    #----------------------------------------------------------------------------------------------------------------------------

    fig3D = plt.figure(1)
    ax = fig3D.add_subplot(111, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')

    ax.scatter(optimized_X[0, :], optimized_X[1, :], optimized_X[2, :], marker='o', color='b')
    ax.scatter(threeDPoints_0[:, 0], threeDPoints_0[:, 1],threeDPoints_0[:, 2], marker='o', color='r')

    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    print('Close the figure to continue.')
    plt.show()

    #-------------------------------------------------------------------------------------------------------------------------

    Unique2DPoints = []

    for i in range(len(kp1_6_match)):
        for a in range(len(common_kp1_old_6)):
            if np.all(kp1_6_match[i] !=  common_kp1_old_6[a]):
                Unique2DPoints.append(kp1_6_match[i])
        
    Unique2DPoints = np.array(Unique2DPoints)

    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display Image ref with points
    ax1.imshow(Image, cmap='gray')
    ax1.plot(np.array(Unique2DPoints)[:, 0], np.array(Unique2DPoints)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(Unique2DPoints), 'r', (10, 0))  # Plot numbered points
    ax1.set_title('Image ref')

    # Display Image 6 with points
    ax2.imshow(image0, cmap='gray')
    ax2.plot(np.array(kp2_0_match)[:, 0], np.array(kp2_0_match)[:, 1], 'rx', markersize=1)
    plotNumberedImagePoints(np.array(kp2_0_match), 'r', (10, 0))  # Plot numbered points
    ax2.set_title('Image 0')
    plt.show()

    #-------------------------------------------------------------------------------------------------------------------------

    scale_w = 1000 / image0.shape[1]  # Width scale factor
    scale_h = 750 / image0.shape[0]   # Height scale factor
    scale = min(scale_w, scale_h)        # Choose the smaller scale to avoid distortion

    # Resize the old photo with the calculated scale while preserving aspect ratio
    new_width = int(image0.shape[1] * scale)
    new_height = int(image0.shape[0] * scale)
    resized_old = cv2.resize(image0, (new_width, new_height))

    # Pad the resized old photo to match the size of the new photo (1000x750)
    top = (750 - new_height) // 2
    bottom = 750 - new_height - top
    left = (1000 - new_width) // 2
    right = 1000 - new_width - left

    # Add padding (black border) around the resized old photo
    padded_old = cv2.copyMakeBorder(resized_old, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    diff_image = background_subtraction_simple(Image, padded_old)
    cv2.imshow("Background Subtraction", diff_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # result_image, diff_image = image_difference(Image, padded_old)
    # cv2.imshow("Differences", result_image)
    # cv2.imshow("Difference Mask", diff_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # score, diff_image = calculate_ssim(Image, padded_old)
    # print("SSIM Score:", score)
    # cv2.imshow("Difference Map", diff_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    edges = edge_detection(Image)
    # Display the result
    cv2.imshow('Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges0 = edge_detection(image0)
    cv2.imshow('Edge Detection', edges0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges0_colored = colorize_edges(edges, (255, 0, 0))  # Blue color (BGR)
    edges1_colored = colorize_edges(edges0, (0, 0, 255))  # Red color (BGR)

    cv2.imshow('Edge Detection', edges0_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Edge Detection', edges1_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # img0 = cv2.imread('Resultados/Edges_blueRot.png')  # Replace with the actual file paths
    # img1 = cv2.imread('Resultados/Edges_red.png')  # Replace with the actual file paths

    # # Ensure the images have the same size
    # if img0.shape[:2] != img1.shape[:2]:
    #     img0 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))

    # # Perform edge detection on both images
    # edges0 = edge_detection(img0)
    # edges1 = edge_detection(img1)

    # # Colorize the edges (blue for the first image, red for the second)
    # edges0_colored = colorize_edges(edges0, (255, 0, 0))  # Blue color (BGR)
    # edges1_colored = colorize_edges(edges1, (0, 0, 255))  # Red color (BGR)

    # # Overlay the two images (using alpha blending to show differences clearly)
    # overlay = cv2.addWeighted(edges0_colored, 0.5, edges1_colored, 0.5, 0)

    # # Display the result
    # cv2.imshow('Edge Detection Overlay', overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

