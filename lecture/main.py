import cv2
import numpy as np


def sign_lines(img_edges) -> np.ndarray:
    lines = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=10, maxLineGap=15)
    return lines

def main() -> None:
        green_sign = cv2.imread("snek.jpg")
        green_sign_hsv = cv2.cvtColor(green_sign, cv2.COLOR_BGR2HSV)
        # print(green_sign_hsv[50:60, 50:60])
        color_low = np.array([30, 20, 115])
        color_high = np.array([57, 70, 205])
        mask = cv2.inRange(green_sign_hsv, color_low, color_high)
        edges = cv2.Canny(mask, 50, 150)

        lines = sign_lines(edges)
        # If lines are detected, draw them on the original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(green_sign, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("detected_lines", green_sign)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        pass

if __name__ == '__main__':
    main()
