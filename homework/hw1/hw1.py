import cv2
import numpy as np
from PyQt5.QtCore import center


def read_image(image_path: str) -> np.ndarray:
    """
    This function reads an image and returns it as a numpy array
    :param image_path: String of path to file
    :return img: Image array as ndarray
    """
    # Read in the image
    image = cv2.imread(image_path)

    # Create a new array
    image_array = np.array(image)

    return image_array


def extract_green(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just green channel
    """
    # Create a new array
    green_img = np.zeros_like(img)

    # Isolate the green channel
    green_img[:, :, 1] = img[:, :, 1]

    return green_img


def extract_red(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the red channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just red channel
    """
    # Create a new array
    red_img = np.zeros_like(img)

    # Isolate the red channel
    red_img[:, :, 2] = img[:, :, 2]

    return red_img


def extract_blue(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the blue channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just blue channel
    """
    # Create a new array
    blue_img = np.zeros_like(img)

    # Isolate the blue channel
    blue_img[:, :, 0] = img[:, :, 0]

    return blue_img


def swap_red_green_channel(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the image with the red and green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of red and green channels swapped
    """
    # Create a new array
    swap_img = np.zeros_like(img)

    # Swap the red and blue channels
    swap_img = img[:, :, [2, 1, 0]]

    return swap_img


def embed_middle(img1: np.ndarray, img2: np.ndarray, embed_size: (int, int)) -> np.ndarray:
    """
    This function takes two images and embeds the embed_size pixels from img2 onto img1
    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :param embed_size: Tuple of size (width, height)
    :return: Image array as ndarray of img1 with img2 embedded in the middle
    """
    # Create new arrays
    img1 = np.array(img1)
    img2 = np.array(img2)

    # Store image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    w, h = embed_size

    # Calculate top left corner of img2 inside img1
    top_left_x1 = (w1 - w) // 2
    top_left_y1 = (h1 - h) // 2

    # Extract the middle of img2
    center_x2 = w2 // 2
    center_y2 = h2 // 2
    top_left_x2 = center_x2 - w // 2
    top_left_y2 = center_y2 - h // 2
    middle_img2 = img2[top_left_y2:top_left_y2 + h, top_left_x2:top_left_x2 + w]

    # Embed extracted img2 into img1
    img1[top_left_y1:top_left_y1 + h, top_left_x1:top_left_x1 + w] = middle_img2

    return img1


def calc_stats(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the mean and standard deviation
    :param img: Image array as ndarray
    :return: Numpy array with mean and standard deviation in that order
    """
    # Create a new array
    img1 = np.array(img)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Calculate the mean and standard deviation
    mean, std_dev = cv2.meanStdDev(img_gray)

    # Make the results more readable
    mean = mean[0][0]
    std_dev = std_dev[0][0]
    arr = np.array([mean, std_dev])

    return arr


def shift_image(img: np.ndarray, shift_val: int) -> np.ndarray:
    """
    This function takes an image and returns the image shifted by shift_val pixels to the right.
    Should have an appropriate border for the shifted area:
    https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    Returned image should be the same size as the input image.
    :param img: Image array as ndarray
    :param shift_val: Value to shift the image
    :return: Shifted image as ndarray
    """
    raise NotImplementedError


def difference_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    This function takes two images and returns the first subtracted from the second

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :return: Image array as ndarray
    """
    raise NotImplementedError


def add_channel_noise(img: np.ndarray, channel: int, sigma: int) -> np.ndarray:
    """
    This function takes an image and adds noise to the specified channel.

    Should probably look at randn from numpy

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img: Image array as ndarray
    :param channel: Channel to add noise to
    :param sigma: Gaussian noise standard deviation
    :return: Image array with gaussian noise added
    """
    raise NotImplementedError


def add_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and adds salt and pepper noise.

    Must only work with grayscale images
    :param img: Image array as ndarray
    :return: Image array with salt and pepper noise
    """
    raise NotImplementedError


def blur_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    This function takes an image and returns the blurred image

    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    :param img: Image array as ndarray
    :param ksize: Kernel Size for medianBlur
    :return: Image array with blurred image
    """
    raise NotImplementedError
