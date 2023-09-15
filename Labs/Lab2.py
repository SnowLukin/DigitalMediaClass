import cv2
import numpy as np


def start_point():
    # Task 1: Read the image from the camera and translate it into HSV format.
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Task 2: Apply image filtering using the command inRange
        # and leave only the red part.
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_only = cv2.bitwise_and(frame, frame, mask=mask)

        # Task 3: Perform morphological transformations of the filtered image.
        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(mask, kernel, iterations=1)
        # dilation = cv2.dilate(mask, kernel, iterations=1)

        image_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        image_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Task 4: Find the moments in the resulting image of the first order,
        # find the area of the object.
        moments = cv2.moments(mask)
        area = moments['m00']

        # Task 5: Based on the analysis of the area of the object,
        # find its center and build a black rectangle around the object.
        if area > 0:
            width = height = int(np.sqrt(area))
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])

            cv2.rectangle(
                frame,
                (c_x - (width // 16), c_y - (height // 16)),
                (c_x + (width // 16), c_y + (height // 16)),
                (0, 0, 0),
                2
            )

        # Display the resulting frames
        cv2.imshow('frame', frame)
        cv2.imshow('Red Only', red_only)
        cv2.imshow("Opening", image_opening)
        cv2.imshow("Closing", image_closing)
        # cv2.imshow('Erosion', erosion)
        # cv2.imshow('Dilation', dilation)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Custom implementation of erode method
#   erode: This method reduces the size of the foreground object and erodes
#  the boundaries of the object's pixels. It removes pixels on the object boundaries
#  and shrinks the image. Erosion is useful for tasks like noise reduction and
#  eliminating small objects or details from an image.
#  The erode method takes the input image and a kernel as parameters.
#  The kernel is a small matrix that defines the neighborhood of each pixel.
#  The method replaces each pixel with the minimum value of its neighborhood, effectively eroding the object.
def erode(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded


# Custom implementation of dilate method
# dilate: This method increases the size of the foreground object and expands
# the boundaries of the object's pixels.
# It adds pixels to the object boundaries and makes it appear larger.
# Dilation is useful for tasks like filling holes, joining broken parts of an object,
# or thickening lines in an image.
# The dilate method also takes the input image and a kernel as parameters.
# It replaces each pixel with the maximum value of its neighborhood, effectively dilating the object.
def dilate(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated
