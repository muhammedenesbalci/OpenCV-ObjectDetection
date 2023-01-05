"""
- Used to recognize a single image
- Point feature matching
"""
import cv2
import matplotlib.pyplot as plt

# Transfer in
img = cv2.imread("chocolates.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

imgFind = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(imgFind, cmap = "gray"),plt.axis("off")

# Orb descriptive (detect edge, corner etc.) (bad)
orb = cv2.ORB_create()

# key point detection
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(imgFind, None)

# Brut Force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Match the points
matches = bf.match(des1, des2)

# sort of the distances
matches = sorted(matches, key = lambda x: x.distance)

# matches points
img_match = cv2.drawMatches(img, kp1, imgFind, kp2, matches[:20], None, flags = 2)
plt.figure(), plt.imshow(img_match), plt.axis("off"),plt.title("orb")

# sift
sift = cv2.xfeatures2d.SIFT_create()

# Brut Force matching
bf = cv2.BFMatcher()

# Key point detection
kp1, des1 = sift.detectAndCompute(imgFind, None)
kp2, des2 = sift.detectAndCompute(img, None)

# Match the points
matches = bf.knnMatch(des1, des2, k=2)

sift_matches_arr = []

for match1, match2 in matches:

    if match1.distance < 0.75 * match2.distance:
        sift_matches_arr.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(imgFind, kp1, img, kp2, sift_matches_arr, None, flags=2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")


plt.show()
