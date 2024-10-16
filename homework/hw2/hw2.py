import cv2
import numpy as np


def sign_sides(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of contours.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of vertices.
    """
    img_cp = img.copy()

    # Convert to grayscale and apply Gaussian blur
    img_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 1.5)

    # Edge detection
    edges = cv2.Canny(img_blur, 50, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=50, maxLineGap=15) # minLineLength=50

    # Detect contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sign_circle(img: np.ndarray, min_radius: int, max_radius: int) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :param min_radius: Minimum radius of circles to detect
    :param max_radius: Maximum radius of circles to detect
    :return: Numpy array of circles.
    """
    img_cp = img.copy()

    # Convert to grayscale and apply Gaussian blur
    img_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 2)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    return circles


def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    # Initialize arrays to hold x and y coordinates
    x_coords = []
    y_coords = []

    # Iterate over the lines to extract coordinates
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Append the midpoints of the lines to the respective lists
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            x_coords.append(mid_x)
            y_coords.append(mid_y)

    # Convert lists to numpy arrays
    return np.array(x_coords, dtype=np.int32), np.array(y_coords, dtype=np.int32)


def show_image(title: str, img: np.ndarray, wait: int = 0):
    """
    Display an image in a window and wait for a key press to continue.
    :param title: The title of the window.
    :param img: The image to display.
    :param wait: Time to wait for a key press (0 for default indefinite wait).
    """
    cv2.imshow(title, img)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


def identify_traffic_light(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple identifying the location
    of the traffic light in the image and the lighted light.
    :param img: Image as numpy array
    :return: Tuple identifying the location of the traffic light in the image and light.
             ( x,   y, color)
             (140, 100, 'None') or (140, 100, 'Red')
             In the case of no light lit, coordinates can be just center of traffic light
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define gray color range
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])

    # Create a mask for gray
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect circles
    circles = sign_circle(masked_img, 10, 30)

    if circles is not None:
        # Convert circle coordinates to integers
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            x, y, radius = circle[0], circle[1], circle[2]

            mask = np.zeros_like(img_cp)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)
            masked_img = cv2.bitwise_and(img_cp, mask)

            avg_color = cv2.mean(masked_img, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

            # Check for more of the desired average color in the traffic light region
            # ([0] for blue channel, [1] for green, [2] for red)
            if avg_color[1] > 150 and avg_color[2] > 150:
                return x, y, 'Yellow'
            elif avg_color[2] > 150:
                return x, y, 'Red'
            elif avg_color[1] > 150:
                return x, y, 'Green'

    return 0, 0, 'None'


def identify_stop_sign(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'stop')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 150, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect contours in the masked area
    contours = sign_sides(masked_img)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the shape is an octagon (8 sides)
        if len(approx) == 8:
            # Get the bounding box to calculate center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Check area size to filter out small contours
            # and check for convexity
            if cv2.contourArea(contour) > 500:
                if cv2.isContourConvex(approx):
                    return center_x, center_y, 'stop'

    return 0, 0, 'None'


def identify_yield(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'yield')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask_red1 | mask_red2

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect contours in the masked area
    contours = sign_sides(masked_img)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the shape is a triangle (3 sides)
        if len(approx) == 3:
            # Get the bounding box to calculate center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Check area size to filter out small contours
            if cv2.contourArea(contour) > 500:
                return center_x, center_y, 'yield'

    return 0, 0, 'None'


def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define orange color range
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    # Create a mask for orange
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect contours in the masked area
    contours = sign_sides(masked_img)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the shape is a diamond-like (4 sides)
        num_sides = len(approx)
        if num_sides == 4 and cv2.isContourConvex(approx):
            # Get the bounding box to calculate center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Check area size to filter out small contours
            if cv2.contourArea(contour) > 500:
                return center_x, center_y, 'construction'

    return 0, 0, 'None'


def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect contours in the masked area
    contours = sign_sides(masked_img)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the shape is diamond-like (4 sides)
        num_sides = len(approx)
        if num_sides == 4 and cv2.isContourConvex(approx):
            # Get the bounding box to calculate center
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Check area size to filter out small contours
            # and check for diamond-like aspect ratio (roughly square-shaped)
            if 0.8 <= aspect_ratio <= 1.2:
                if cv2.contourArea(contour) > 500:
                    return x + w // 2, y + h // 2, 'warning'

    return 0, 0, 'None'


def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'rr_crossing')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # show_image('', masked_img)

    # Detect contours in the masked area
    circles = sign_circle(masked_img, 50, 100)

    if circles is not None:
        # Convert circles to integer values
        circles = np.round(circles[0, :]).astype("int")

        # Get the (x, y) coordinates and radius of the detected circle
        for (x, y, r) in circles:
            return x, y, 'rr_crossing'

    return 0, 0, 'None'


def identify_services(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'services')
    """
    img_cp = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([100, 150, 25])
    upper_blue = np.array([140, 255, 255])

    # Create a mask for blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (0, 0), 1.5)

    # Find contours on the masked image
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours to find rectangular shapes
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Check if the approximated contour has 4 sides (rectangle)
        if len(approx) == 4:
            # Get the bounding box to calculate center
            x, y, w, h = cv2.boundingRect(approx)
            center_x = x + w // 2
            center_y = y + h // 2

            # Check area size to filter out small contours
            if cv2.contourArea(contour) > 500:
                return center_x, center_y, 'services'

    # Return 'None' if no service signs are detected
    return 0, 0, 'None'


def identify_signs(img: np.ndarray) -> list[list]:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: List of lists of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    img_cp = img.copy()

    # List to store the detected signs
    found_signs = []

    # List to store the function names
    detection_funcs = [
        identify_construction, identify_stop_sign, identify_yield, identify_rr_crossing, identify_services,
        identify_warning, identify_traffic_light
    ]

    for func in detection_funcs:
        x, y, sign_name = func(img_cp)
        if sign_name != 'None':
            found_signs.append([x, y, sign_name])

    return found_signs


def identify_signs_noisy(img: np.ndarray) -> list[list]:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will have gaussian noise applied to them so you will need to do some blurring before detection.
    :param img: Image as numpy array
    :return: List of lists of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    img_cp = img.copy()

    # Blur the image
    img_blur = cv2.GaussianBlur(img_cp, (0, 0), 3.5)

    # List to store the detected signs
    found_signs = []

    # List to store the function names
    detection_funcs = [
        identify_construction, identify_stop_sign, identify_yield, identify_rr_crossing, identify_services,
        identify_warning, identify_traffic_light
    ]

    for func in detection_funcs:
        x, y, sign_name = func(img_blur)
        if sign_name != 'None':
            found_signs.append([x, y, sign_name])

    return found_signs


def identify_signs_real(img: np.ndarray) -> list[list]:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will be real images so you will need to do some preprocessing before detection.
    You may also need to adjust existing functions to detect better with real images through named parameters
    and other code paths

    :param img: Image as numpy array
    :return: List of lists of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    img_cp = img.copy()

    # List to store the detected signs
    found_signs = []

    # List to store the function names
    detection_funcs = [
        identify_construction, identify_stop_sign, identify_yield, identify_rr_crossing, identify_services,
        identify_warning
    ] # identify_traffic_light

    for func in detection_funcs:
        x, y, sign_name = func(img_cp)
        if sign_name != 'None':
            found_signs.append([x, y, sign_name])

    return found_signs
