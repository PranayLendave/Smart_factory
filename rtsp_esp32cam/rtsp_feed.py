''' Youtube Channel : Asim Code
View RTSP Stream from IP Camera, Live video streaming over the network with OpenCV in Python
https://youtu.be/ZBRqWPE-_yM
'''

import cv2
import imutils
from imutils.video import VideoStream

rtsp_url = "rtsp://192.168.43.222:8554/mjpeg/1"
video_stream = VideoStream(rtsp_url).start()

while True:
    frame = video_stream.read()
    if frame is None:
        continue

    frame = imutils.resize(frame,width=480)
    cv2.imshow('RTSP_feed', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video_stream.stop()