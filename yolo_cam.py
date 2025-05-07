import cv2, torch, platform, time

model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, verbose=False)
model.conf = 0.3                     # raise if you get many false positives
model.iou  = 0.45
model.to('cpu')                      # Pi 3 has no CUDA
model.eval()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # PiCam or USB cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)   # keep frames small
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

t0 = time.time(); frames = 0
while True:
    ret, img = cap.read()
    if not ret: break
    res  = model(img, size=320)
    cv2.imshow("YOLOv5n @320px - q to quit", res.render()[0])
    frames += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print(f"{frames/(time.time()-t0):.2f} FPS on", platform.uname().machine)
cap.release() ; cv2.destroyAllWindows()
