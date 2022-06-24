import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("opencv_frame_15.png")
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
plt.figure(figsize=(7,5))
plt.title("res")
plt.imshow(res)

# grey_1=cv2.cvtColor(res,cv2.Color)
grey_1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

(thresh, blackAndWhiteImage) = cv2.threshold(grey_1, 10, 255, cv2.THRESH_BINARY)

cv2.imshow('Black white image', blackAndWhiteImage)
cv2.waitKey(0)
invert = cv2.bitwise_not(blackAndWhiteImage)
cv2.imshow('Inverted image', invert)
cv2.waitKey(0)

plt.show()
# cv2.destroyAllWindows()