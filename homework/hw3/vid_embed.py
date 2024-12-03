import cv2
import numpy as np

# Set up ArUco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

marker_layout = {
    "top_left": 0,
    "top_right": 1,
    "bottom_left": 2,
    "bottom_right": 3
}

def process_video(main_video_path, embed_video_path, output_video_path):
    # Open the main video and the embedded video
    main_video = cv2.VideoCapture(main_video_path)
    embed_video = cv2.VideoCapture(embed_video_path)

    # Get video properties
    fps = int(main_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(main_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(main_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    embed_fps = int(embed_video.get(cv2.CAP_PROP_FPS))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Read a frame from both videos
        ret_main, main_frame = main_video.read()
        ret_embed, embed_frame = embed_video.read()

        if not ret_main:
            break  # Exit loop if the main video ends

        # If embed video ends, loop it
        if not ret_embed:
            embed_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_embed, embed_frame = embed_video.read()

        # Detect markers in the main frame
        corners, ids, _ = detector.detectMarkers(main_frame)

        if ids is not None and len(ids) >= 4:
            marker_centers = {id[0]: corner[0].mean(axis=0) for corner, id in zip(corners, ids)}

            if all(marker_layout[key] in marker_centers for key in marker_layout):
                # Arrange source and destination points
                src_points = np.array(
                    [marker_centers[marker_layout[key]] for key in
                     ["top_left", "top_right", "bottom_right", "bottom_left"]],
                    dtype=np.float32
                )

                embed_height, embed_width = embed_frame.shape[:2]
                dst_points = np.array([
                    [0, embed_height - 1],
                    [embed_width - 1, embed_height - 1],
                    [embed_width - 1, 0],
                    [0, 0]
                ], dtype=np.float32)

                # Compute the homography
                H, _ = cv2.findHomography(dst_points, src_points)

                # Warp the embed frame
                warped_embed = cv2.warpPerspective(embed_frame, H, (frame_width, frame_height))

                # Create mask and blend
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, src_points.astype(int), 255)

                masked_main = cv2.bitwise_and(main_frame, main_frame, mask=cv2.bitwise_not(mask))
                main_frame = cv2.add(masked_main, warped_embed)

        # Write the frame to the output video
        out_video.write(main_frame)

    # Release resources
    main_video.release()
    embed_video.release()
    out_video.release()

main_video_path = "vid_reference/20241203_144737.mp4"
embed_video_path = "vid_embed/backrooms.mp4"
output_video_path = "output/20241203_144737.mp4"
process_video(main_video_path, embed_video_path, output_video_path)
