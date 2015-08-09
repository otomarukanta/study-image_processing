import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_amplitude(amplitude):
    abs_amplitude = np.absolute(amplitude)
    norm_amplitude = np.log(abs_amplitude + 1)
    norm_amplitude = norm_amplitude / np.max(norm_amplitude)
    return norm_amplitude


def circle_mask(img, r):
    rows, cols = img.shape
    y, x = np.mgrid[-rows/2:rows/2, -cols/2:cols/2]
    circle = x ** 2 + y ** 2
    mask = np.zeros(img.shape, dtype=bool)
    mask[circle < r ** 2] = True

    return mask


def low_pass(amplitude, r):
    dst = amplitude
    mask = circle_mask(dst, r)
    dst[np.logical_not(mask)] = 0

    return dst


def high_pass(amplitude, r):
    dst = amplitude
    mask = circle_mask(dst, r)
    dst[mask] = 0

    return dst

if __name__ == "__main__":
    img = cv2.imread("../dataset/4.2.04.tiff", 0)

    fourier = np.fft.fft2(img)
    f_shift = np.fft.fftshift(fourier)

    lowpass_amplitude = low_pass(f_shift, 50)

    f_ishift = np.fft.ifftshift(lowpass_amplitude)
    lowpass_img = np.fft.ifft2(f_ishift)
    lowpass_img = np.absolute(lowpass_img)
    lowpass_img = lowpass_img / np.max(lowpass_img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.imshow(lowpass_img, cmap='gray')
    plt.show()
