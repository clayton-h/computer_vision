import cv2
import numpy as np
import glob

# Prepare object points (3D points in real-world space)
checkerboard_size = (9, 6)  # Number of inner corners per row and column
square_size = 0.025  # Length of one square side in meters (adjust for your checkerboard)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('calibration_images/*.jpg')  # Path to your calibration images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform camera calibration
ret, cam_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration results
print("Camera matrix:\n", cam_matrix)
print("Distortion coefficients:\n", dist_coeffs)

np.save("camera_matrix.npy", cam_matrix)
np.save("dist_coeffs.npy", dist_coeffs)
