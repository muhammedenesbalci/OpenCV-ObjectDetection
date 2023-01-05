"""
* Match method is a method to search and find the position of a template image in a larger image
* we shift the template image over the master image and compare the area occupied by the template
image and the template image on the master image
* A comparison is made of how similar the matchup is at each location
"""

import cv2
import matplotlib.pyplot as plt

pathCat = "D:\\Githubbb\\4.ObjectDetection\\5.TemplateMatching\\imgs\\cat.jpg"
pathCatFace = "D:\\Githubbb\\4.ObjectDetection\\5.TemplateMatching\\imgs\\cat_face.jpg"

# Transfer in
imgCat = cv2.imread(pathCat, 0)
print(imgCat.shape)
plt.figure(), plt.imshow(imgCat, cmap="gray"), plt.axis("off"), plt.title("Original")

imgCatFace = cv2.imread(pathCatFace, 0)
(h, w) = imgCatFace.shape  # attention here !! it is not(w, h) it is (h, w)
print(imgCatFace.shape)
plt.figure(), plt.imshow(imgCatFace, cmap="gray"), plt.axis("off"), plt.title("Template")

# Matchup
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method = eval(meth)  # Convert strings to a function

    res = cv2.matchTemplate(imgCat, imgCatFace, method)
    print(meth, " : ", res.shape)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(imgCat, top_left, bottom_right, 255, 2)

    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(res, cmap="gray"), plt.axis("off")
    plt.subplot(1, 2, 2), plt.imshow(imgCat, cmap="gray"), plt.axis("off")
    plt.suptitle("method : {}".format(meth))

plt.show()
