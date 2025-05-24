import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    cx = int(width / 2)
    cy = int(height / 2)

    pixel_hsv = hsv_frame[cy, cx]
    h, s, v = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

    color = "Undefined"
    if (h < 10 or h > 170) and s > 100 and v > 50:
        color = "MERAH"
    elif 10 <= h <= 35 and s > 100:
        color = "KUNING"
    elif 36 <= h <= 85 and s > 100:
        color = "HIJAU"
    elif 86 <= h <= 130 and s > 100:
        color = "BIRU"
    elif 131 <= h <= 160 and s > 100:
        color = "UNGU"
    elif s < 50 and v > 180:
        color = "PUTIH"
    elif v < 50:
        color = "HITAM"

    bgr_pixel = frame[cy, cx]
    b, g, r = int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])

    # Tampilkan warna dan nilai HSV
    cv2.putText(frame, f"{color}", (cx - 200, 100), 0, 3, (b, g, r), 5)
    cv2.putText(frame, f"H:{h} S:{s} V:{v}", (cx - 200, 50), 0, 1, (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
