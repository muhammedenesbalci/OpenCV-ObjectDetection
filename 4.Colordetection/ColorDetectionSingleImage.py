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

# HSV Bounds in blue in image
lower_hsv = (84, 98, 80)
higher_hsv = (184, 255, 255)

# read img
frame = cv2.imread("img.jpg")
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# mask blue color
masked = cv2.inRange(frame_hsv, lower_hsv, higher_hsv)
cv2.imshow("Masked", masked)

# Morphology operations
eroding_img = cv2.erode(masked, None, iterations=2)
cv2.imshow("after eroding_img", eroding_img)

dilation_img = cv2.dilate(eroding_img, None, iterations=2)
cv2.imshow("after dilation_img", dilation_img)

# find contours
(contours, _) = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # take the bigger are within contours
    maxContour = max(contours, key=cv2.contourArea)

    # Get the rect (min area)
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

    cv2.imshow("result", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
