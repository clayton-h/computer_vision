import os
import cv2
from hw1 import *


def main() -> None:
    # DONE: Add in images to read
    img1 = read_image("hw1_pic1.jpg")
    img2 = read_image("hw1_pic2.jpg")

    # DONE: replace None with the correct code to convert img1 and img2
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    img1_red = extract_red(img1)
    img1_green = extract_green(img1)
    img1_blue = extract_blue(img1)

    img2_red = extract_red(img2)
    img2_green = extract_green(img2)
    img2_blue = extract_blue(img2)

    img1_swap = swap_red_green_channel(img1)
    img2_swap = swap_red_green_channel(img2)

    embed_img = embed_middle(img1, img2, (60, 60))

    img1_stats = calc_stats(img1_gray)
    img2_stats = calc_stats(img2_gray)

    # DONE: Replace None with correct calls
    img1_shift = shift_image(img1_gray, 2)
    img2_shift = shift_image(img2_gray, 2)

    img1_diff = difference_image(img1_gray, img1_shift)
    img2_diff = difference_image(img2_gray, img2_shift)

    # TODO: Select appropriate sigma and call functions
    sigma = 15
    img1_noise_red = add_channel_noise(img1, 2, sigma)
    img1_noise_green = add_channel_noise(img1, 1, sigma)
    img1_noise_blue = add_channel_noise(img1, 0, sigma)

    img2_noise_red = add_channel_noise(img2, 2, sigma)
    img2_noise_green = add_channel_noise(img2, 1, sigma)
    img2_noise_blue = add_channel_noise(img2, 0, sigma)

    #
    # Why are some channels more affected visually by noise than others?
    #
    # Apparently, the human eye is more sensitive to green than other colors.
    # Green noise is thereby more noticeable than red or blue noise.
    #
    # This is exasperated by the fact that green contributes the most to the
    # perceived brightness and detail of an image.
    #
    # Not to mention the contents of the images themselves;
    # images with more prominent color channels will appear noisier.
    #

    img1_spnoise = add_salt_pepper(img1_gray)
    img2_spnoise = add_salt_pepper(img2_gray)

    # TODO: Select appropriate ksize, must be odd
    # ksize = 0
    # img_blur = blur_image(img1_spnoise, ksize)
    # img2_blur = blur_image(img2_spnoise, ksize)

    # TODO: Write out all images to appropriate files
    cv2.imwrite("hw1_pic1_gray.jpg", img1_gray)
    cv2.imwrite("hw1_pic2_gray.jpg", img2_gray)

    cv2.imwrite("hw1_pic1_hsv.jpg", img1_hsv)
    cv2.imwrite("hw1_pic2_hsv.jpg", img2_hsv)

    cv2.imwrite("hw1_pic1_red.jpg", img1_red)
    cv2.imwrite("hw1_pic2_red.jpg", img2_red)

    cv2.imwrite("hw1_pic1_green.jpg", img1_green)
    cv2.imwrite("hw1_pic2_green.jpg", img2_green)

    cv2.imwrite("hw1_pic1_blue.jpg", img1_blue)
    cv2.imwrite("hw1_pic2_blue.jpg", img2_blue)

    cv2.imwrite("hw1_pic1_swap.jpg", img1_swap)
    cv2.imwrite("hw1_pic2_swap.jpg", img2_swap)

    cv2.imwrite("hw1_embedded.jpg", embed_img)

    print(img1_stats)
    print(img2_stats)

    cv2.imwrite("hw1_pic1_shifted.jpg", img1_shift)
    cv2.imwrite("hw1_pic2_shifted.jpg", img2_shift)

    cv2.imwrite("hw1_pic1_difference.jpg", img1_diff)
    cv2.imwrite("hw1_pic2_difference.jpg", img2_diff)

    cv2.imwrite("hw1_pic1_rednoise.jpg", img1_noise_red)
    cv2.imwrite("hw1_pic1_greennoise.jpg", img1_noise_green)
    cv2.imwrite("hw1_pic1_bluenoise.jpg", img1_noise_blue)
    cv2.imwrite("hw1_pic2_rednoise.jpg", img2_noise_red)
    cv2.imwrite("hw1_pic2_greennoise.jpg", img2_noise_green)
    cv2.imwrite("hw1_pic2_bluenoise.jpg", img2_noise_blue)

    cv2.imwrite("hw1_pic1_spnoise.jpg", img1_spnoise)
    cv2.imwrite("hw1_pic2_spnoise.jpg", img2_spnoise)

if __name__ == '__main__':
    main()
