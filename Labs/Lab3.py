import cv2
import numpy as np


def start_point():
    img = cv2.imread("/Users/snowlukin/Desktop/DigitalMediaClass/Resources/lab2_img1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', img)

    deviation = 5
    size = 3
    blur1 = gauss_blur_convolution(img, size, deviation)
    cv2.imshow(f'Size: {size}, Deviation: {deviation}', blur1)

    deviation = 10
    size = 7
    blur2 = gauss_blur_convolution(img, size, deviation)
    cv2.imshow(f'Size: {size}. Deviation: {deviation}', blur2)

    blur_open_cv = cv2.GaussianBlur(img, (size, size), deviation)

    cv2.imshow(f'OpenCV - Size: {size}. Deviation: {deviation}', blur_open_cv)
    cv2.waitKey(0)


def gauss_kernel(x, y, standard_deviation, a, b):
    # a, b – Математическое ожидание двумерной случайной величины
    # gauss[x, y] = 1 / (2 * pi * sd^2) * e^(- ((x-a)^2 + (y - b)^2) / (2 * sd^2))
    standard_deviation_mult = 2 * standard_deviation ** 2
    m1 = 1 / (np.pi * standard_deviation_mult)
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / standard_deviation_mult)
    return m1 * m2


def make_gauss_kernel(size, deviation):
    kernel = np.ones((size, size))
    a = b = (size + 1) // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = gauss_kernel(i, j, deviation, a, b)
    return kernel


def perform_convolution(size, kernel):
    result_sum = 0
    for i in range(size):
        for j in range(size):
            result_sum += kernel[i, j]
    for i in range(size):
        for j in range(size):
            kernel[i, j] /= result_sum
    print(kernel)
    return kernel


def gauss_blur_convolution(img, size, deviation):

    kernel = make_gauss_kernel(size, deviation)

    # print(kernel)

    kernel = perform_convolution(size, kernel)

    blur = img.copy()
    start_x = size // 2
    start_y = size // 2

    for i in range(start_x, blur.shape[0] - start_x):
        for j in range(start_y, blur.shape[1] - start_y):
            blur[i,  j] = calculate_value(img, kernel, i, j)

    return blur


def calculate_value(img, kernel, size, i, j):
    value = 0
    for k in range(-(size // 2), size // 2 + 1):
        for i in range(-(size // 2), size // 2 + 1):
            value += img[i + k, j + i] * kernel[(size // 2) + k, (size // 2) + i]
    return value
