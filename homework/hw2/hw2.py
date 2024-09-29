import cv2
import numpy as np


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """
    # Copy the image
    img_cp = img.copy()

    # Image grayscale conversion and Gaussian blur
    img_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 1.5)

    # Edge and line detection
    img_edge = cv2.Canny(img_blur, 100, 200)
    lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=15)

    return lines


def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """
    # Copy the image
    img_cp = img.copy()

    # Convert the image to grayscale and blur
    img_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 2)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=100
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
    # Copy the image
    img_cp = img.copy()

    # Detect circles
    circles = sign_circle(img_cp)

    if circles is not None:
        # Convert circle coordinates to integers
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            x, y, radius = circle[0], circle[1], circle[2]

            mask = np.zeros_like(img_cp)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=1)
            masked_img = cv2.bitwise_and(img_cp, mask)

            avg_color = cv2.mean(masked_img, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

            # Check for yellow (both red and green channels are high)
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
    # Copy the image
    img_cp = img.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Define the color range for detecting orange
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect sign lines from the masked image
    lines = sign_lines(masked_img)

    if lines is None:
        return 0, 0, 'None'

    # Get x and y coordinates from detected lines
    x, y = sign_axis(lines)

    # Calculate the average position
    if len(x) > 0 and len(y) > 0:
        avg_x = np.mean(x).astype(int)
        avg_y = np.mean(y).astype(int)

        return avg_x, avg_y, 'stop'

    return 0, 0, 'None'


def identify_yield(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'yield')
    """
    # # Copy the image
    # img_cp = img.copy()
    #
    # # Convert the image to HSV
    # hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)
    #
    # # Lower red hue (0-10 degrees)
    # lower_red1 = np.array([0, 100, 100])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([170, 100, 100])
    # upper_red2 = np.array([180, 255, 255])
    #
    # # Define the color range for detecting red
    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # mask = cv2.bitwise_or(mask1, mask2)
    #
    # # Apply the mask to the image
    # masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)
    #
    # # Detect sign lines from the masked image
    # lines = sign_lines(masked_img)
    #
    # if lines is None:
    #     return 0, 0, 'None'
    #
    # # Get x and y coordinates from detected lines
    # x, y = sign_axis(lines)
    #
    # # Calculate the average position
    # if len(x) > 0 and len(y) > 0:
    #     avg_x = np.mean(x).astype(int)
    #     avg_y = np.mean(y).astype(int)
    #
    #     return avg_x, avg_y, 'yield'

    return 0, 0, 'None'


def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    # Copy the image
    img_cp = img.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting orange
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([30, 255, 255])

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect sign lines
    lines = sign_lines(masked_img)

    if lines is None:
        return 0, 0, 'None'

    # Get x and y coordinates from detected lines
    x, y = sign_axis(lines)

    # Calculate the average position
    if len(x) > 0 and len(y) > 0:
        avg_x = np.mean(x).astype(int)
        avg_y = np.mean(y).astype(int)

        # Return the detected sign center coordinates and label
        return avg_x, avg_y, 'construction'

    return 0, 0, 'None'



def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    # Copy the image
    img_cp = img.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting yellow
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([60, 255, 255])

    # Define the color range for detecting yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect sign lines from the masked image
    lines = sign_lines(masked_img)

    if lines is None:
        return 0, 0, 'None'

    # Get x and y coordinates from detected lines
    x, y = sign_axis(lines)

    # Calculate the average position
    if len(x) > 0 and len(y) > 0:
        avg_x = np.mean(x).astype(int)
        avg_y = np.mean(y).astype(int)

        return avg_x, avg_y, 'warning'

    return 0, 0, 'None'


def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'rr_crossing')
    """
    # Copy the image
    img_cp = img.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Define the color range for detecting yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect circles
    circles = sign_circle(masked_img)

    if circles is not None:
        # Convert circles to integer values
        circles = np.round(circles[0, :]).astype("int")

        # Get the (x, y) coordinates and radius of the first detected circle
        for (x, y, r) in circles:
            # Check for sign characteristics, e.g., circle size range
            if 20 <= r <= 100:
                # Return the center coordinates of the detected circle
                return x, y, 'rr_crossing'

    return 0, 0, 'None'


def identify_services(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'services')
    """
    # Copy the image
    img_cp = img.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting blue
    lower_blue = np.array([230, 200, 250])
    upper_blue = np.array([245, 255, 255])

    # Define the color range for detecting white (for the border)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Define the color range for detecting blue and white
    mask_blue= cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_blue, mask_white)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_cp, img_cp, mask=mask)

    # Detect sign lines from the masked image
    lines = sign_lines(masked_img)

    if lines is None:
        return 0, 0, 'None'

    # Get x and y coordinates from detected lines
    x, y = sign_axis(lines)

    # Calculate the average position
    if len(x) > 0 and len(y) > 0:
        avg_x = np.mean(x).astype(int)
        avg_y = np.mean(y).astype(int)

        return avg_x, avg_y, 'services'

    return 0, 0, 'None'


def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    # Copy the image
    img_cp = img.copy()

    # Initialize an empty list to store the detected signs
    found_signs = []

    # Call sign detection function
    x, y, sign_name = identify_construction(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_stop_sign(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_yield(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_rr_crossing(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_services(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_warning(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    return found_signs


def identify_signs_noisy(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will have gaussian noise applied to them so you will need to do some blurring before detection.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    # Copy the image
    img_cp = img.copy()

    # Blur the image
    img_blur = cv2.GaussianBlur(img_cp, (0, 0), 1.5)

    # Initialize an empty list to store the detected signs
    found_signs = []

    # Call sign detection function
    x, y, sign_name = identify_construction(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_stop_sign(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_yield(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_rr_crossing(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_services(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_warning(img_blur)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    return found_signs


def identify_signs_real(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will be real images so you will need to do some preprocessing before detection.
    You may also need to adjust existing functions to detect better with real images through named parameters
    and other code paths

    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    # Copy the image
    img_cp = img.copy()

    # Initialize an empty list to store the detected signs
    found_signs = []

    # Call sign detection function
    x, y, sign_name = identify_construction(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_stop_sign(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_yield(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_rr_crossing(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_services(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    # Call sign detection function
    x, y, sign_name = identify_warning(img_cp)
    if sign_name != 'None':
        found_signs.append([x, y, sign_name])

    return found_signs
