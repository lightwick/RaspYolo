import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 320)}))
picam2.start()

frames = 0

while True:
    img = picam2.capture_array()
    cv2.imshow("YOLOv5n @320px - q to quit", img)
    frames += 1
    cv2.waitKey(25)
cv2.destroyAllWindows()