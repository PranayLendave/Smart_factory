# -------- Libraries--------------
import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
from datetime import datetime
import time
import os

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
Current_seconds = 0
Previous_seconds = 0
Current_seconds1 = 0
Previous_seconds1 = int(round(time.time()))
sort = 0
first_state = 1
img = '0'
contours_1 = '0'
jar_clone = '0'
img_name = '0'
img_file = '0'


def creating_files():
    path = os.getcwd()
    path_1 = path + '/Original_Images'
    isFile = os.path.isdir(path_1)
    print(isFile)
    if not isFile:
        path_1 = path + '/Original_Images'
        os.mkdir(path_1)
        print('Original_Images is missing [Created!!]')

    path_1 = path + '/Cropped_Images'
    isFile = os.path.isdir(path_1)
    if not isFile:
        path_1 = path + '/Cropped_Images'
        os.mkdir(path_1)
        print('Cropped_Images is missing [Created!!]')

    path_1 = path + '/Detected_Images'
    isFile = os.path.isdir(path_1)
    if not isFile:
        path_1 = path + '/Detected_Images'
        os.mkdir(path_1)
        path_2 = path_1 + "/Under_filled"
        os.mkdir(path_2)
        path_2 = path_1 + "/Over_filled"
        os.mkdir(path_2)
        path_2 = path_1 + "/Perfect_filled"
        os.mkdir(path_2)
        print('Detected_Images is missing [Created!!]')


def data_logging(state_):
    global now
    global dt_string
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open("Data_Logging.txt", "a") as f:
        f.write(f"Level detected is {state_},Time: {dt_string}\n")


def capture():
    try:
        global now
        image_name1 = "0"
        image_file1 = "0"
        webcam = 0  # 1 - for external webcam
        cam = cv2.VideoCapture(webcam, cv2.CAP_DSHOW)
        img_counter = 0

        if cam.isOpened():
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("Level_Detection", frame)

                # k = cv2.waitKey(1)
                # if k % 256 == 27:
                #     # ESC pressed
                #     print("Escape hit, closing...")
                #     break
                # elif k % 256 == 32:
                #     # SPACE pressed
                #     image_name = "img_{}.png".format(utc_timestamp)
                #     image_file = "Original_Images/" + image_name
                #     cv2.imwrite(image_file, frame)
                #     print("{} written!".format(image_name))
                #     img_counter += 1

                cv2.waitKey(2000)
                if img_counter < 1:
                    image_name1 = "img_{}.png".format(float("{:.2f}".format(time.time())))
                    image_file1 = "Original_Images/" + image_name1
                    cv2.imwrite(image_file1, frame)
                    print("{} written!".format(image_name1))
                    img_counter += 1
                else:
                    break

            cam.release()
            cv2.destroyAllWindows()
            return "Connected", image_file1, image_name1
        else:
            print("Alert ! Camera disconnected")
            return "Disconnected", "Empty", "Empty"

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass


def level_detection_code():
    global jar_clone
    global img_file
    global img_name
    global img

    status, image__file, image__name = capture()
    img_name = image__name
    img_file = image__file

    # ====================
    # img_path = "Original_Images/"  # Load the image and Resize it
    # img_name = "opencv_frame_15.png"
    # img_file = img_path + img_name
    # ====================

    # ==========Cropping image===========
    img = cv2.imread(img_file)
    dimension = img.shape
    # width = int(dimension[1]/2)
    # height = int(dimension[0]/2)
    # img = cv2.resize(img, (width, height))

    print("Shape of the image", dimension)  # Shape of the image
    r1, r2 = 0, 430
    c1, c2 = 140, 490
    # [rows, columns]
    img = img[r1:r2, c1:c2]
    # cv2.imshow("IMG in level_detecion",img)
    # cv2.imshow('Cropped_Image', img)
    cv2.imwrite("Cropped_Images/" + "Cropped_" + img_name, img)
    # cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # ===========================================

    # ============= Colour detection ==============
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image color from BGR to RGB
    # plt.figure(figsize=(7,5))
    # plt.title("img")
    # plt.imshow(grid_RGB)

    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)  # Convert the image color from RGB to HSV

    lower = np.array([0, 100, 50])  # Set the color range to be detected
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(grid_HSV, lower, upper)  # Masking the image
    # plt.figure(figsize=(7,5))
    # plt.imshow(mask)

    res = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask)
    # plt.figure(figsize=(7,5))
    # plt.title("res")
    # plt.imshow(res)

    # grey_1=cv2.cvtColor(res,cv2.Color)
    grey_1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    (thresh, blackAndWhiteImage) = cv2.threshold(grey_1, 10, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Black white image', blackAndWhiteImage)
    # cv2.waitKey(0)
    inverted_image = cv2.bitwise_not(blackAndWhiteImage)
    # cv2.imshow('inverted_imageed image', inverted_image)
    # cv2.waitKey(0)
    # ===============================================

    # read image and take first channel only
    jar_channel = inverted_image.copy()

    # draw histogram
    y, x, _ = plt.hist(jar_channel.ravel(), 256, [0, 256])

    plt.show()

    # manual threshold
    (T, jar_threshold) = cv2.threshold(jar_channel, 50, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Bottle Gray Threshold", jar_threshold)
    # cv2.waitKey(0)

    # apply opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    jar_open = cv2.morphologyEx(jar_threshold, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Bottle Open 5 x 5", jar_open)
    # cv2.waitKey(0)

    # find all contours
    contours = cv2.findContours(jar_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    jar_clone = jar_channel.copy()
    img_1 = img.copy()
    # print(type(img))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    cv2.drawContours(img_1, contours, -1, (255, 0, 0), 2)
    # cv2.imshow("All Contours", img_1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # sort contours by area

    areas = [cv2.contourArea(contour) for contour in contours]
    # print(areas)
    (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a: a[1]))
    global contours_1
    contours_1 = contours
    # print contour with largest area
    jar_clone = jar_channel.copy()
    img_2 = img.copy()
    # cv2.imshow("img for largest contrours", img)
    cv2.drawContours(img_2, [contours[-1]], -1, (255, 0, 0), 2)
    # cv2.imshow("Largest contour", img_2)
    # cv2.waitKey(0)

    # draw bounding box, calculate aspect and display decision

    jar_clone = img.copy()
    (x, y, w, h) = cv2.boundingRect(contours[-1])
    aspectRatio = w / float(h)
    aspectRatio = 1 / aspectRatio
    aspectRatio = round(aspectRatio, 3)
    print(aspectRatio)
    return aspectRatio


def decision_block():
    sort = 0
    state = ""
    jar_clone = img.copy()

    # cv2.imshow("img in decision block",img)
    (x, y, w, h) = cv2.boundingRect(contours_1[-1])
    if aspectRatio > 0.9:
        cv2.rectangle(jar_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(jar_clone, "Over filled, cf={value:.3f}".format(value=aspectRatio, ),
                    (x + 10, y + 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        state = "Over_filled"
        print("Overfilled queatity: ", aspectRatio - 0.92)
        sort = 1
        data_logging(state)
    elif 0.77 < aspectRatio < 0.92:
        cv2.rectangle(jar_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(jar_clone, "Perfect, cf={value:.3f}".format(value=aspectRatio, ), (x + 10, y + 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        state = "Perfect_filled"
        data_logging(state)

    elif aspectRatio < 0.77:
        cv2.rectangle(jar_clone, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(jar_clone, "Under filled,{value:.3f}".format(value=aspectRatio), (x + 10, y + 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        state = "Under_filled"
        print("Underfilled queatity: ", 0.77 - aspectRatio)
        sort = 1
        data_logging(state)

    cv2.imshow("Decision", jar_clone)
    cv2.imwrite(f"Detected_Images/{state}/" + "Detected_" + img_name, jar_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sort


if __name__ == '__main__':
    creating_files()
    while True:
        try:
            Current_seconds = float("{:.2f}".format(time.time()))
            if Current_seconds - Previous_seconds > 10:
                print("Entered the loop: .............")
                Previous_seconds = Current_seconds
                aspectRatio = level_detection_code()  # getting aspectRatio value from the function
                sort = decision_block()  # getting decision for sorting from the decision block

            if sort == 1:
                if first_state == 1:
                    Previous_seconds1 = int(round(time.time()))
                first_state = 0
                Current_seconds1 = int(round(time.time()))
                if Current_seconds1 - Previous_seconds1 > 5:
                    Previous_seconds = Current_seconds
                    print("Sort the jar")
                    sort = 0

        except KeyboardInterrupt:
            print("[Closed!]KeyboardInterrupt occurred")
            exit()
        except ValueError:
            print("No bottle detected")
