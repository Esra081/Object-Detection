from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")


model = YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

limits = [400, 297, 490, 297, 590, 297, 673, 297]

while True:

    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1


            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass =="truck" or currentClass == "bus"\
                    or currentClass == "motorbike" and conf >0.3:
                cvzone.putTextRect(imgGray, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,
                                   offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=5)
                currentArray= np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))  # stack

    resultsTracker = tracker.update(detections)  # put the bounding box variables in array
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255),5)
    cv2.line(img, (limits[2], limits[3]), (limits[4], limits[5]), (0, 255, 255),5)
    cv2.line(img, (limits[4], limits[5]), (limits[6], limits[7]), (255, 0, 255),5)



    for result in resultsTracker:
        x1, y1, x2, y2, Id = map(int, result)
        print(result)
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=5, rt=3, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=3)

    cx, cy = x1+w//2, y1+h//2
    cv2.circle(img,(cx, cy), 5, (255, 0, 255), cv2.FILLED)

    if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20:
            totalCount.append(Id)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    if limits[2] < cx < limits[4] and limits[3] - 20 < cy < limits[5] + 20:
            totalCount.append(Id)
            cv2.line(img, (limits[2], limits[3]), (limits[4], limits[5]), (0, 255, 0), 5)

    if limits[4] < cx < limits[6] and limits[5] - 20 < cy < limits[7] + 20:
            totalCount.append(Id)
            cv2.line(img, (limits[4], limits[5]), (limits[6], limits[7]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN,5, (255, 0, 0), 8)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
