import cv2, torch, platform, time
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 320)}))
picam2.start()

model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, verbose=False)
model.conf = 0.3                     # raise if you get many false positives
model.iou  = 0.45
model.to('cpu')                      # Pi 3 has no CUDA
model.eval()

t0 = time.time(); frames = 0
while True:
    img = picam2.capture_array()
    res  = model(img, size=320)
    cv2.imshow("YOLOv5n @320px - q to quit", res.render()[0])
    frames += 1
    cv2.waitKey(25)

#print(f"{frames/(time.time()-t0):.2f} FPS on", platform.uname().machine)
cv2.destroyAllWindows()
