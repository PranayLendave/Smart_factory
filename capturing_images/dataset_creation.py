#safe copy for programme dont edit here
#captring the image using inbuilt cam and saving at desired path with appropriate sub folders 

import cv2
import os
import time
import uuid

images_path = 'E:\\PRANAY\\CLG\\6th sem\\ANNFL\\IA\\New folder\\RealTimeObjectDetection-main\\Tensorflow\\workspace\\images\\allimages'
#images_path='E:\\PRANAY\\CLG\\6th sem\\ANNFL\\IA\\New folder\\hand_gesture_IA\\test_images'
os.chdir(images_path)
#os.makedirs('tutoria_1')
labels = ['a', 'b']
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
number_imgs = 15

for label in labels:
    cap = cv2.VideoCapture(0)
    print('collecting images for {}'.format(label))
    time.sleep(7)
    os.makedirs(label)
    
    for imgnum_1 in range(number_imgs):
        ret, frame = cap.read()
        imgnum=label+'\\'+'image_'+label+'_'+str(imgnum_1)+'.jpg'
        cv2.imwrite(imgnum, frame)
        cv2.imshow('live transmission', frame)
        print('image_'+label+'_'+str(imgnum_1))
        time.sleep(4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()