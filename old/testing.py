import math

import cv2
from ultralytics import YOLO

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("./runs/detect/train3/weights/best.pt")

# object classes
classNames = [
    "10c",
    "10d",
    "10h",
    "10s",
    "2c",
    "2d",
    "2h",
    "2s",
    "3c",
    "3d",
    "3h",
    "3s",
    "4c",
    "4d",
    "4h",
    "4s",
    "5c",
    "5d",
    "5h",
    "5s",
    "6c",
    "6d",
    "6h",
    "6s",
    "7c",
    "7d",
    "7h",
    "7s",
    "8c",
    "8d",
    "8h",
    "8s",
    "9c",
    "9d",
    "9h",
    "9s",
    "Ac",
    "Ad",
    "Ah",
    "As",
    "Jc",
    "Jd",
    "Jh",
    "Js",
    "Kc",
    "Kd",
    "Kh",
    "Ks",
    "Qc",
    "Qd",
    "Qh",
    "Qs",
]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
