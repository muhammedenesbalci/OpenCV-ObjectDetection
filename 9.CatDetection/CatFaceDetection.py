import cv2
import os

path = "D:\\Githubbb\\4.ObjectDetection\\9.CatDetection\\images\\"
files = os.listdir(path)
print(files)

img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
print(img_path_list)

for j in img_path_list:
    print(j)

    image = cv2.imread(path + j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor=1.045, minNeighbors=2)

    """
    The enumerate function in Python converts a data collection object into 
    an enumerate object. Enumerate returns an object 
    that contains a counter as a key for each value within an object, 
    making items within the collection easier to access.
    """

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, "Kedi {}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow(j, image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue
