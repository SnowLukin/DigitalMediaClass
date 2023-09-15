import cv2
import numpy as np
from Utils import InputUtils


# TASK 1
def task1():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# TASK 2
def task2():
    img1 = cv2.imread('/Users/snowlukin/Desktop/DigitalMediaClass/Resources/lab2_img1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/Users/snowlukin/Desktop/DigitalMediaClass/Resources/lab2_img2.png', cv2.IMREAD_REDUCED_COLOR_8)
    img3 = cv2.imread('/Users/snowlukin/Desktop/DigitalMediaClass/Resources/lab2_img3.webp', cv2.IMREAD_ANYDEPTH)
    cv2.namedWindow('img1', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('img2', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TASK 3
def task3():
    cap = cv2.VideoCapture('/Resources/screen_vid.mp4', cv2.CAP_ANY)

    w = 1280
    h = 720

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_HLS2RGB)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


# TASK 4
def task4():
    video = cv2.VideoCapture('/Resources/screen_vid.mp4', cv2.CAP_ANY)
    _, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    video_writer = cv2.VideoWriter(
        "/Resources/Output/screen_vid_copy.mp4",
        fourcc,
        25,
        (w, h)
    )
    while True:
        ok, vid = video.read()
        if not ok:
            break

        # cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# TASK 5
def task5():
    img1 = cv2.imread('/Users/snowlukin/Desktop/DigitalMediaClass/Resources/lab2_img3.webp')

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img_hsv', cv2.WINDOW_NORMAL)

    cv2.imshow('img', img1)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    cv2.imshow('img_hsv', hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TASK 6
def task6():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        cross_image = np.zeros((height, width, 3), dtype=np.uint8)

        vertical_line_width = 60
        vertical_line_height = 300
        cv2.rectangle(cross_image,
                      (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                      (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                      (0, 125, 255), 2)

        horizontal_line_width = 250
        horizontal_line_height = 55
        cv2.rectangle(cross_image,
                      (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                      (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                      (0, 125, 255), 2)

        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

        cv2.imshow("video", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# TASK 7
def task7():
    video = cv2.VideoCapture(0)
    _, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    video_writer = cv2.VideoWriter(
        "/Users/snowlukin/Desktop/DigitalMediaClass/Resources/Output/recorded_video.mp4",
        fourcc,
        25,
        (w, h)
    )

    while True:
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


# TASK 8
def task8():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        cross_image = np.zeros((height, width, 3), dtype=np.uint8)

        vertical_line_width = 60
        vertical_line_height = 300
        cv2.rectangle(cross_image,
                      (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                      (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                      (0, 0, 255), 2)
        rect_start_v = (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2)
        rect_end_v = (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2)

        horizontal_line_width = 250
        horizontal_line_height = 55
        cv2.rectangle(cross_image,
                      (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                      (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                      (0, 0, 255), 2)

        rect_start_h = (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2)
        rect_end_h = (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2)

        central_pixel_color = frame[height // 2, width // 2]

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = image_hsv.shape
        central_pixel = image_hsv[height // 2, width // 2]

        hue = central_pixel[0]  # assuming central_pixel is in HSV color space

        if (hue > 0) and (hue < 30) or (150 <= hue <=180):  # closest to red if hue is between 0-30 or 150-180
            print('Red')
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 0, 255), -1)
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 0, 255), -1)
        elif (30 <= hue < 90):  # closest to green if hue is between 30-90
            print('Green')
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 255, 0), -1)
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 255, 0), -1)
        else:  # closest to blue if hue is between 90-150
            print('Blue')
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (255, 0, 0), -1)
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (255, 0, 0), -1)

        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

        cv2.imshow("video", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# # TASK 9
# def task9():
#     cap = cv2.VideoCapture(0)
#
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#
#         cv2.imshow("camera", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


# Starting point
def start_point():
    task_number_input = input('Enter task number: ')

    if not InputUtils.is_int(task_number_input):
        print('Wrong format')
        exit()

    task_number = int(task_number_input)

    match task_number:
        case 1:
            task1()
        case 2:
            task2()
        case 3:
            task3()
        case 4:
            task4()
        case 5:
            task5()
        case 6:
            task6()
        case 7:
            task7()
        case 8:
            task8()
        # case 9:
        #     task9()
        case _:
            print(f'Task{task_number} doesnt exist')
            exit()
