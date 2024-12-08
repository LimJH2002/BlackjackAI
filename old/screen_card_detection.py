import math
import cv2
from ultralytics import YOLO
import numpy as np
from mss import mss # pip install mss
import win32gui # pip install pywin32
import win32con
import win32ui
import win32api

# YOLO model
model = YOLO("./runs/detect/train3/weights/best.pt")

# Object classes
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

# Screen capture setup
sct = mss()
monitor = sct.monitors[1]  # Fullscreen capture; replace with specific region if needed

# Get screen's device context
hwnd = win32gui.GetDesktopWindow()
hdc = win32gui.GetWindowDC(hwnd)
hdc_draw = win32ui.CreateDCFromHandle(hdc)
pen = win32ui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 255))  # Purple bounding box
hdc_draw.SelectObject(pen)

while True:
    # Capture screen
    screenshot = np.array(sct.grab(monitor))

    # Convert to BGR for compatibility with YOLO
    img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Run detection
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            label = f"{classNames[cls]} {confidence:.2f}"

            # Draw bounding box on screen
            hdc_draw.MoveTo(x1, y1)
            hdc_draw.LineTo(x1, y2)
            hdc_draw.LineTo(x2, y2)
            hdc_draw.LineTo(x2, y1)
            hdc_draw.LineTo(x1, y1)

            # Display label
            hdc_draw.TextOut(x1 + 5, y1 - 20, label)