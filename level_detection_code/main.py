import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np


img=cv2.imread("opencv_frame_15.png")
img=cv2.resize(img,(600,800))
grid_RGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(7,5))
# plt.title("img")
# plt.imshow(grid_RGB)

grid_HSV=cv2.cvtColor(grid_RGB,cv2.COLOR_RGB2HSV)

lower=np.array([0,150,50])
upper=np.array([20,255,255])

mask= cv2.inRange(grid_HSV,lower ,upper)
# plt.figure(figsize=(7,5))
# plt.imshow(mask)

res = cv2.bitwise_and(grid_RGB, grid_RGB,mask=mask)
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

##===============================================
# read image and take first channel only
bottle_3_channel = invert
bottle_3_channel=cv2.resize(bottle_3_channel,(600,800))
bottle_gray = cv2.split(bottle_3_channel)[0]
# cv2.imshow("Bottle Gray", bottle_gray)
# cv2.waitKey(0)


# blur image
bottle_gray = cv2.GaussianBlur(bottle_gray, (7, 7), 0)
# cv2.imshow("Bottle Gray Smoothed 7 x 7", bottle_gray)
# cv2.waitKey(0)

# draw histogram
y,x,_ =plt.hist(bottle_gray.ravel(), 256,[0, 256]);
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
(contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
# print contour with largest area
bottle_clone = bottle_3_channel.copy()
cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
# cv2.imshow("Largest contour", bottle_clone)
# cv2.waitKey(0)


# draw bounding box, calculate aspect and display decision
bottle_clone = img
(x, y, w, h) = cv2.boundingRect(contours[-1])
aspectRatio = w / float(h)
print(1/aspectRatio)
if aspectRatio > 0.9:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(bottle_clone, "Over filled, cf={value:.3f}".format(value=1/aspectRatio,), (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

elif 0.8 < aspectRatio < 0.9:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "80% filled, cf={value:.3f}".format(value=1 / aspectRatio, ), (x + 10, y + 20),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    # data_logging()
elif 0.8 < aspectRatio < 0.9:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "Perfect, cf={value:.3f}".format(value=1 / aspectRatio, ), (x + 10, y + 20),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    # data_logging(data_1)
elif 0.7 < aspectRatio < 0.8:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "70% filled, cf={value:.3f}".format(value=1 / aspectRatio, ), (x + 10, y + 20),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    # data_logging(data_1)
elif 0.6 < aspectRatio < 0.7:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "60% filled, cf={value:.3f}".format(value=1 / aspectRatio, ), (x + 10, y + 20),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    # data_logging(data_1)
else:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.putText(bottle_clone, "Under filled,{value:.3f}".format(value=1/aspectRatio), (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    # data_logging(data_1)
cv2.imshow("Decision", bottle_clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
# def data_logging(data_1):
#     with open("Data_Logging.txt", "a") as f:
#         f.write(f"Jar detected is {data_1},Time: {dt_string}\n")