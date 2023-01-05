import cv2

cascade_path = "cascade.xml"
frameWidth = 240
frameHeight = 240

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

color = (255, 0, 0)


def empty(a): pass


# trackbar
cv2.namedWindow("Sonuc")
cv2.resizeWindow("Sonuc", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Sonuc", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Sonuc", 4, 50, empty)

# cascade classifier
cascade = cv2.CascadeClassifier("cascade.xml")

while True:

    # read img
    success, img = cap.read()

    if success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detection parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Sonuc") / 1000)
        neighbor = cv2.getTrackbarPos("Neighbor", "Sonuc")
        # detectiom
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)

        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, "object", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

        cv2.imshow("Sonuc", img)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindow()
