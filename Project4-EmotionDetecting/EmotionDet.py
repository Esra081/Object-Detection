from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) # for webcam
cap.set(3, 640)  # width
cap.set(4, 480)  # height

model = YOLO("best.pt")
classNames = ['0', '1', '2', '3']

while True:

    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1

            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = round(conf*100, 2)

            conf, cls = map(lambda x: x[0], (box.conf, box.cls))  # her ikisinden de güven skoru en yüksek olanı aldık
            cls = map(int(box.cls[0]))

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()