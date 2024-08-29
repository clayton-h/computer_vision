import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.GimpGradientFile import linear


def main() -> None:
    window = "BlurExample"
    img = cv2.imread('snek.jpg')

    blurred_image = cv2.GaussianBlur(img, (3, 3), 1.5)

    cv2.imshow(window, img)
    cv2.imshow(window+"2", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    main()