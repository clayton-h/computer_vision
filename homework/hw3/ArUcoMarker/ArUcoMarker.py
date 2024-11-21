import cv2
import cv2.aruco as aruco

# Choose the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Marker size in pixels
marker_size = 200

# IDs of the markers to generate
marker_ids = [0, 1, 2, 3]

for marker_id in marker_ids:
    # Generate the marker
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker to a file
    file_name = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(file_name, marker_image)
    print(f"Marker ID {marker_id} saved as {file_name}")

    # Optionally, display each marker
    cv2.imshow(f"Marker {marker_id}", marker_image)
    cv2.waitKey(500)  # Display each marker for 500ms
cv2.destroyAllWindows()
