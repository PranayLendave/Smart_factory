# -------- Libraries--------------
import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
from datetime import datetime
import time

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

Current_seconds = 0
Previous_seconds = 0
Current_seconds1 = 0
Previous_seconds1 = int(round(time.time()))
sort = 0
first_state = 1


def data_logging(state_):
    global now
    global dt_string
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open("Data_Logging.txt", "a") as f:
        f.write(f"Level detected is {state_},Time: {dt_string}\n")


def capture():
    try:
        global now
        webcam = 1  # 1 - for external webcam
        cam = cv2.VideoCapture(webcam)
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
                    image_name = "img_{}.png".format(float("{:.2f}".format(time.time())))
                    image_file = "Original_Images/" + image_name
                    cv2.imwrite(image_file, frame)
                    print("{} written!".format(image_name))
                    img_counter += 1
                else:
                    break

            cam.release()
            cv2.destroyAllWindows()
            return "Connected", image_file, image_name
        else:
            print("Alert ! Camera disconnected")
            return "Disconnected", "Empty", "Empty"

    except KeyboardInterrupt:
        # print("KeyboardInterrupt")
        pass


while True:
    try:
        Current_seconds = float("{:.2f}".format(time.time()))
        if Current_seconds - Previous_seconds > 20:
            print("Entered the loop: .............")
            Previous_seconds = Current_seconds
            print(Previous_seconds)
            # Some work to be done
            status, image__file, image__name = capture()
            # print(status)
            # print(image__file)
            # print(image__name)
            img_name = image__name
            img_file = image__file

            # ---------
            # ====================
            # Load the image and Resize it
            # img_path = "Original_Images/"
            # img_name = "opencv_frame_15.png"
            # img_file = img_path + img_name
            #====================

            img = cv2.imread(img_file)
            dimension = img.shape
            # width = int(dimension[1]/2)
            # height = int(dimension[0]/2)
            # img = cv2.resize(img, (width, height))

            # Shape of the image
            print("Shape of the image", dimension)

            r1, r2 = 0, 430
            c1, c2 = 140, 490

            # [rows, columns]
            img = img[r1:r2, c1:c2]

            cv2.imshow('Cropped_Image', img)
            cv2.imwrite("Cropped_Images/" + "Cropped_" + img_name, img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            # Convert the image color from BGR to RGB
            grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.figure(figsize=(7,5))
            # plt.title("img")
            # plt.imshow(grid_RGB)

            # Convert the image color from RGB to HSV
            grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

            # Set the color range to be detected
            lower = np.array([0, 150, 50])
            upper = np.array([20, 255, 255])

            # Masking the image
            mask = cv2.inRange(grid_HSV, lower, upper)
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
            invert = cv2.bitwise_not(blackAndWhiteImage)
            # cv2.imshow('Inverted image', invert)
            # cv2.waitKey(0)

            # ===============================================

            # read image and take first channel only
            bottle_3_channel = invert
            bottle_gray = cv2.split(bottle_3_channel)[0]
            # cv2.imshow("Bottle Gray", bottle_gray)
            # cv2.waitKey(0)

            # blur image
            bottle_gray = cv2.GaussianBlur(bottle_gray, (7, 7), 0)
            # cv2.imshow("Bottle Gray Smoothed 7 x 7", bottle_gray)
            # cv2.waitKey(0)

            # draw histogram
            y, x, _ = plt.hist(bottle_gray.ravel(), 256, [0, 256]);
            # print(x[np.where(y==)])
            # print(y.max())
            # plt.show()

            # manual threshold
            (T, bottle_threshold) = cv2.threshold(bottle_gray, 50, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("Bottle Gray Threshold", bottle_threshold)
            # cv2.waitKey(0)

            # apply opening operation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bottle_open = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("Bottle Open 5 x 5", bottle_open)
            # cv2.waitKey(0)

            # find all contours
            contours = cv2.findContours(bottle_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            bottle_clone = bottle_3_channel.copy()
            cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)
            # cv2.imshow("All Contours", bottle_clone)
            # cv2.waitKey(0)

            # sort contours by area
            areas = [cv2.contourArea(contour) for contour in contours]
            (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a: a[1]))
            # print contour with largest area
            bottle_clone = bottle_3_channel.copy()
            cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
            # cv2.imshow("Largest contour", bottle_clone)
            # cv2.waitKey(0)

            # draw bounding box, calculate aspect and display decision
            bottle_clone = img
            (x, y, w, h) = cv2.boundingRect(contours[-1])
            aspectRatio = w / float(h)
            print(1 / aspectRatio)
            state = ""
            if aspectRatio > 0.9:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(bottle_clone, "Over filled, cf={value:.3f}".format(value=1 / aspectRatio, ),
                            (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                state = "Over_filled"
                sort = 0
                data_logging(state)
            elif 0.8 < aspectRatio < 0.9:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(bottle_clone, "80% filled, cf={value:.3f}".format(value=1 / aspectRatio, ),
                            (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                state = "80%_filled"
                data_logging(state)
            elif 0.8 < aspectRatio < 0.9:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(bottle_clone, "Perfect, cf={value:.3f}".format(value=1 / aspectRatio, ), (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                state = "Perfect_filled"
                data_logging(state)
            elif 0.7 < aspectRatio < 0.8:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(bottle_clone, "70% filled, cf={value:.3f}".format(value=1 / aspectRatio, ),
                            (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                state = "70%_filled"
                data_logging(state)
            elif 0.6 < aspectRatio < 0.7:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(bottle_clone, "60% filled, cf={value:.3f}".format(value=1 / aspectRatio, ),
                            (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                state = "60%_filled"
                data_logging(state)
            else:
                cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(bottle_clone, "Under filled,{value:.3f}".format(value=1 / aspectRatio), (x + 10, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                state = "Under_filled"
                data_logging(state)

            cv2.imshow("Decision", bottle_clone)
            cv2.imwrite(f"Detected_Images/{state}/" + "Detected_" + img_name, bottle_clone)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

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
        # print("KeyboardInterrupt")
        pass


