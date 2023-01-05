"""
RGB-HSW
RED, GREEN, BLUE --- HUE, SATURATION, VALUE
HUE --> Represent Colors
Saturation --> Turkish mean is 'Doygunluk'
Value --> Brightness
"""
"""
HSV values (I used Paint on windows os)
- Browse hsv ranges of individual points of the object
HSV --> (184, 64, 67)
Lower --> (84, 98, 0)
Upper --> (184, 255, 255)
"""

import cv2
import numpy as np

# Capture
cap = cv2.VideoCapture(1)
cap.set(3, 960)  # width
cap.set(4, 480)  # height

lower_hsv = (84, 98, 80)
higher_hsv = (184, 255, 255)

while True:
    success, frame = cap.read()

    if success:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # masking with hsv
        masked = cv2.inRange(frame_hsv, lower_hsv, higher_hsv)

        # Morphology operations
        eroding_img = cv2.erode(masked, None, iterations=2)

        dilation_img = cv2.dilate(eroding_img, None, iterations=2)
        cv2.imshow("Video", eroding_img)

        (contours, _) = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # take the bigger are within contours
            maxContour = max(contours, key=cv2.contourArea)

            # Get the rect
            rect = cv2.minAreaRect(maxContour)
            ((x, y), (width, height), rotation) = rect  # x y is the centers points

            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)

            # Draw rect
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 3)

            # Draw center
            M = cv2.moments(maxContour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
