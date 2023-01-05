import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer image in
img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.title("Original")

# Detect Edges 0-255
img_edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.figure(), plt.imshow(img_edges, cmap="gray"), plt.title("Edges 0-255")

# Detect Edges th with median some useful constants
median_value = np.median(img)
low = int(max(0, (1 - 0.33) * median_value))
high = int(max(0, (1 + 0.33) * median_value))

img_median = cv2.Canny(image=img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(img_median, cmap="gray"), plt.title("Edges th as median")

# With Blurring
blurred_img = cv2.blur(img, ksize=(4, 4))
median_value = np.median(img)

low = int(max(0, (1 - 0.33) * median_value))
high = int(max(0, (1 + 0.33) * median_value))

img_median = cv2.Canny(image=blurred_img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(img_median, cmap="gray"), plt.title("Edges th as median and blurring")

plt.show()
