import cv2
import numpy as np
import os

# Create the ArUco detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Marker size in meters
marker_length = 0.05

# Set up the coordinate system (no camera calibration, so no pose estimation)
obj_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# Options
estimate_pose = False  # Disable pose estimation
show_rejected = False  # Toggle rejected marker visualization

# Process a single image or multiple images
def process_static_images(image_path):
    for image_path in image_path:
        print(f"Processing {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(image)

        # Draw detected markers
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)

            # If pose estimation was enabled, you could draw axes here, but now we skip that
            # Since estimate_pose is False, we do not calculate or draw axes

        # Draw rejected markers if enabled
        if show_rejected and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

        # Display the results
        cv2.imshow("Processed Image", image_copy)
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == 27:  # Exit if ESC is pressed
            break

    cv2.destroyAllWindows()


# Example usage
image_folder = "img_reference"  # Replace with folder path or empty string

# Load images from folder if folder is provided, otherwise process a single image
if image_folder:
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
else:
    single_image_path = "0.jpg"  # Replace with single image path
    image_paths = [single_image_path]

process_static_images(image_paths)
