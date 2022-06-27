import cv2

img = cv2.imread("img0.png")
print(type(img))

# Shape of the image
print("Shape of the image", img.shape)

# [rows, columns]
crop = img[0:430, 140:490]

# cv2.imshow('original', img)
cv2.imshow('cropped', crop)
cv2.imwrite("crop.png", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
