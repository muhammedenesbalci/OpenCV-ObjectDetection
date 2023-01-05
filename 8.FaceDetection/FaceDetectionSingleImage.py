import cv2
import matplotlib.pyplot as plt

# Transfer in
einstein = cv2.imread("images\\einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

# Classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x, y, w, h) in face_rect:
    cv2.rectangle(einstein, (x, y), (x + w, y + h), (255, 255, 255), 10)

plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

# Multi Faces
# Transfer in
barce = cv2.imread("images\\barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap="gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(barce, minNeighbors=7)

"""
it looks at how many matching pictures are there at the same spot and 
then says there's a face there you
you can try neighbor parameter
"""

for (x, y, w, h) in face_rect:
    cv2.rectangle(barce, (x, y), (x + w, y + h), (255, 255, 255), 10)
plt.figure(), plt.imshow(barce, cmap="gray"), plt.axis("off")
plt.show()
