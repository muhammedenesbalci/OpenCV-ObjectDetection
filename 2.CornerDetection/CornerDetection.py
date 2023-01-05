import cv2
import matplotlib.pyplot as plt
import numpy as np

# transfer image in
img = cv2.imread("sudoku.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off"), plt.title("Original")

# Harris Corner Detection
corners_img = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
plt.figure(), plt.imshow(corners_img, cmap="gray"), plt.axis("off"), plt.title("Harris Corner Detection")

# Dilation, remember morphology lessons it can increase white areas
kernel = np.ones((3, 3), dtype=np.uint8)
dilation_img = cv2.dilate(corners_img, kernel, iterations=1)
plt.figure(), plt.imshow(dilation_img, cmap="gray"), plt.axis("off"), plt.title("Harris Corner Detection with dilation")

# Shi Tomasi Detection
corners_img = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)  # (img, max corner, quality, min distance between corners)
img_corners = img
print("img : ", img_corners)
print("Corners : ", corners_img)  # Float
corners_img = np.int64(corners_img)  # convert to int

for i in corners_img:
    x, y = i.ravel()  # array to tuple
    cv2.circle(img_corners, (x, y), 5, (0, 0, 255), cv2.FILLED)
plt.figure(), plt.imshow(img_corners, cmap="gray"), plt.axis("off"), plt.title("Image corners")

plt.show()
