#captring the image using inbuilt cam and saving at desired path
#for esp32 upload how2electronics code

import cv2
import os
import urllib.request
import numpy as np
import time
import uuid
 
images_path = 'E:\\PRANAY\\CLG\\6th sem\\ANNFL\\IA\\New folder\\RealTimeObjectDetection-main\\Tensorflow\\workspace\\images\\allimages'
#images_path='E:\\PRANAY\\CLG\\6th sem\\ANNFL\\IA\\New folder\\hand_gesture_IA\\test_images'
os.chdir(images_path)
# url='http://192.168.0.105/cam-lo.jpg'
#url='http://192.168.0.105/cam-mid.jpg'
url='http://192.168.0.105/cam-hi.jpg'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
 
count=0

while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)


#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cv2.imshow("live transmission", frame)
    
    
    key=cv2.waitKey(10)
    
    if key==ord('k'):
        imgnum='xyz_'+str(count)+'.jpg'
        cv2.imwrite(imgnum, frame)
        print("image saved as: ",count)
        count+=1
        
    if key==ord('q'):
        break
    else:
        continue
  
cap.release() 
cv2.destroyAllWindows()