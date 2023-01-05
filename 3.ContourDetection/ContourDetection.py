"""
-method that aims to connect all continuous points (with the border) of the same color and density
-Analyze shape and article
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer img
img = cv2.imread("contour.jpg", 0)
plt.figure(), plt.imshow(img, "gray"), plt.axis("off"), plt.title("Original")

# Detect Contour
# (img, find internal or external contours, give us the cordinates to corners)
contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    #  External
    if hierarch[0][i][3] == -1:  # it gives us the external contours
        cv2.drawContours(external_contours, contours, i, 255, -1)  # 255 color, -1 fill it
    # Internal
    else:
        cv2.drawContours(internal_contours, contours, i, 255, -1)  # 255 color, -1 fill it

plt.figure(), plt.imshow(external_contours, "gray"), plt.axis("off"), plt.title("External")
plt.figure(), plt.imshow(internal_contours, "gray"), plt.axis("off"), plt.title("Internal")

print(cv2.version)

plt.show()
