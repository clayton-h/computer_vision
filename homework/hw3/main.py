import cv2
import cv2.aruco as aruco
import numpy as np
import time

# Load camera calibration parameters from .npy files
try:
    cam_matrix = np.load("./CameraCalibration/camera_matrix.npy")
    dist_coeffs = np.load("./CameraCalibration/dist_coeffs.npy")
    print("Camera calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: Calibration files not found. Please calibrate the camera first.")
    exit()

# Initialize the detector
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_params)

# Video capture setup
input_video = cv2.VideoCapture()
wait_time = 0

video = ""  # Replace with video path or leave empty
cam_id = 0  # Default camera ID
marker_length = 0.1  # Marker length in meters
estimate_pose = True  # Set to False if pose estimation is not required
show_rejected = True  # Set to False if rejected markers shouldn't be displayed

if video:
    input_video.open(video)
    wait_time = 0
else:
    input_video.open(cam_id)
    wait_time = 10

# Coordinate system for marker pose estimation
obj_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

total_time = 0
total_iterations = 0

while input_video.grab():
    ret, image = input_video.retrieve()
    if not ret:
        break
    image_copy = image.copy()

    start_time = time.time()

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None:
        n_markers = len(corners)
        rvecs = []
        tvecs = []

        if estimate_pose:
            for i in range(n_markers):
                rvec, tvec, _ = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
                rvecs.append(rvec)
                tvecs.append(tvec)

        # Draw detected markers
        aruco.drawDetectedMarkers(image_copy, corners, ids)

        # Draw pose estimation results
        if estimate_pose:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs, rvec, tvec, marker_length * 1.5, 2)

    # Optionally show rejected candidates
    if show_rejected and rejected:
        aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

    # Calculate and display timing
    current_time = time.time() - start_time
    total_time += current_time
    total_iterations += 1

    if total_iterations % 30 == 0:
        print(f"Detection Time = {current_time * 1000:.2f} ms "
              f"(Mean = {total_time * 1000 / total_iterations:.2f} ms)")

    # Display the results
    cv2.imshow("out", image_copy)
    key = cv2.waitKey(wait_time) & 0xFF
    if key == 27:  # Esc key
        break

input_video.release()
cv2.destroyAllWindows()
