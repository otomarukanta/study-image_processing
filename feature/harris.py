import cv2
import numpy as np
import sys


def opencv_harris(img):
    dst = np.float32(img)
    dst = cv2.cornerHarris(dst, 2, 3, 0.06)
    return dst


def my_harris(img, k, sigma):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    dxx = dx * dx
    dxy = dx * dy
    dyy = dy * dy
    sxx = cv2.GaussianBlur(dxx, (3, 3), sigma)
    sxy = cv2.GaussianBlur(dxy, (3, 3), sigma)
    syy = cv2.GaussianBlur(dyy, (3, 3), sigma)
    d = np.array([[sxx, sxy],
                  [sxy, syy]])
    la, v = np.linalg.eig(np.transpose(d, (2, 3, 0, 1)))
    dst = la[:, :, 0] * la[:, :, 1] - k * (la[:, :, 0] + la[:, :, 1]) ** 2
    return dst


if __name__ == "__main__":

    img = cv2.imread("../dataset/4.2.04.tiff")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst1 = opencv_harris(img)
    dst2 = my_harris(img, 0.06, 0.5)

    cv2.imshow("src", img)
    cv2.imshow("opencv", dst1)
    cv2.imshow("my", dst2)
    cv2.waitKey(0)
