import cv2
import numpy as np

def main() -> None:
    img = cv2.imread('snek.jpg')
    # img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    img_edge = cv2.Canny(img_blur, 100, 200)
    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 60)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('img', img)
    cv2.waitKey(0)

    # log_kernel = np.array([[-1, -1, -1],
    #                       [-1, 8, -1],
    #                       [-1, -1, -1]])
    #
    # log_img = cv2.filter2D(img, -1, log_kernel)
    # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # log_img_dilate = cv2.dilate(log_img, shape)
    # log_img_erode = cv2.erode(log_img, shape)

    # cv2.imshow('log_img', log_img)
    # cv2.imshow('log_img_dilate', log_img_dilate)
    # cv2.imshow('log_img_erode', log_img_erode)

    # img_blur = cv2.GaussianBlur(img, (5, 5), 1.5)
    # img_edge = cv2.Canny(img, 75, 175)

    # window = "BlurExample"

    # cv2.imshow('canny', img_edge)
    # cv2.imshow('original', img)
    # cv2.imshow(window+"2", img_blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()