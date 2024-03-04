from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-Weights/yolov8n.pt')  # Download yolov8 from gitHub
results = model('Images/1.png', show=True)
cv2.waitKey(0)