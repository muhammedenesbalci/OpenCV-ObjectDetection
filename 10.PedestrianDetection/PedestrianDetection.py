import cv2
import os

path = "D:\\Githubbb\\4.ObjectDetection\\10.PedestrianDetection\\"
files = os.listdir(path)
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)

print(img_path_list)

# HOG description
hog = cv2.HOGDescriptor()

# SVM
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)

    image = cv2.imread(path + imagePath)

    (rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Detection: ", image)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue
