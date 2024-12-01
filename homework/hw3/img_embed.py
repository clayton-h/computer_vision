import cv2
import numpy as np
import os

# Create the ArUco detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Marker IDs for layout (adjust these IDs based on your marker configuration)
marker_layout = {
    "top_left": 0,
    "top_right": 1,
    "bottom_left": 2,
    "bottom_right": 3
}

def process_static_images(image_paths, embed_image_path):
    # Load the image to embed
    embed_image = cv2.imread(embed_image_path)
    if embed_image is None:
        print(f"Error: Unable to load embed image {embed_image_path}")
        return

    for image_path in image_paths:
        print(f"Processing {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(image)

        if ids is None or len(ids) < 4:
            print("Error: Not all required markers detected.")
            continue

        # Find the center points of the detected markers
        marker_centers = {}
        for corner, id in zip(corners, ids.flatten()):
            center = corner[0].mean(axis=0)
            marker_centers[id] = center

        # Check if all required markers are present
        if not all(marker_layout[key] in marker_centers for key in marker_layout):
            print("Error: Not all layout markers detected.")
            continue

        # Arrange the points for homography
        src_points = np.array(
            [marker_centers[marker_layout[key]] for key in ["top_left", "top_right", "bottom_right", "bottom_left"]],
            dtype=np.float32)

        # Correctly arrange the corners of the embed image
        embed_height, embed_width = embed_image.shape[:2]
        dst_points = np.array([
            [0, embed_height - 1],  # Top-left of the embed image
            [embed_width - 1, embed_height - 1],  # Top-right of the embed image
            [embed_width - 1, 0],  # Bottom-right of the embed image
            [0, 0]  # Bottom-left of the embed image
        ], dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(dst_points, src_points)

        # Warp the embed image to fit within the marker region
        warped_embed = cv2.warpPerspective(embed_image, H, (image.shape[1], image.shape[0]))

        # Create a mask from the warped image
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, src_points.astype(int), 255)

        # Overlay the warped image onto the original
        masked_original = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        result = cv2.add(masked_original, warped_embed)

        # Save the result
        output_path = os.path.join("output", os.path.basename(image_path))
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"Saved processed image to {output_path}")

# Paths
image_folder = "img_reference"
embed_image_path = os.path.join("img_embed", "embed.jpg")  # Replace with the actual embed image name

# Load images from folder
if image_folder:
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
else:
    single_image_path = "0.jpg"  # Replace with single image path
    image_paths = [single_image_path]

# Process images
process_static_images(image_paths, embed_image_path)
