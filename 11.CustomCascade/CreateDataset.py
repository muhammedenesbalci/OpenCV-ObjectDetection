"""
* prepare positive, negative datasets
* download cascade program
* create custom cascade -> cascade.html will eb created in classifier directory
* use cascade

* you have to use constant background like wall
"""

import cv2
import os

# Dataset file path
pathOfDir = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\4.ObjectDetection\\47.CustomCascade\\Dataset"


# Create Dataset Directory
def createDatasetDir():
    numberOfFiles = 0
    while os.path.exists(pathOfDir + str(numberOfFiles)):
        numberOfFiles += 1

    os.mkdir(pathOfDir + str(numberOfFiles))

    return pathOfDir + str(numberOfFiles)


pathOfDir = createDatasetDir()

# Frame Features
frameWidth = 240
frameHeight = 240

cap = cv2.VideoCapture(1)
# cap.set(3, 640)
# cap.set(4, 480)

# Open cam and saving images
count = 0  # we do not want to save each frame
countForNameImages = 0

while True:
    succes, frame = cap.read()

    if succes:
        frame = cv2.resize(frame, (frameWidth, frameHeight))
        if count % 5 == 0:
            cv2.imwrite(pathOfDir + "\\" + "frame_{}".format(countForNameImages) + ".png", frame)
            countForNameImages +=1
            print(countForNameImages)
        count +=1

        cv2.imshow("Frames", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

