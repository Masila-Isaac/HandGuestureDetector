import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Configuration
width, height = 960, 540
folderPath = "presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Load presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
imgNumber = 0

# Webcam display size
ws, hs = int(213 * 1.2), int(120 * 1.2)

# Gesture and annotation settings
gestureThreshold = 350
buttonPressed = False
buttonCounter = 0
buttonDelay = 20
annotations = [[]]
annotationNumber = -1
annotationStart = False

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Load current slide
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (width, height))

    # Gesture threshold line
    cv2.line(img, (0, gestureThreshold),
             (width, gestureThreshold), (0, 255, 0), 5)

    # Detect hands
    hands, img = detector.findHands(img, flipType=False)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = (xVal, yVal)

        # Slide control gestures (when hand is above threshold line)
        if cy <= gestureThreshold:
            annotationStart = False

            # Left gesture
            if fingers == [1, 0, 0, 0, 0]:
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1

            # Right gesture
            elif fingers == [0, 0, 0, 0, 1]:
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1

        # Pointer gesture (index + middle)
        elif fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        # Draw gesture (only index finger)
        elif fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # Erase gesture (index + pinky)
        elif fingers == [0, 1, 0, 0, 1]:
            if annotations:
                annotations = annotations[:-1]
                annotationNumber -= 1
                annotationStart = False
                buttonPressed = True

    # Button delay
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # Draw annotations
    for annotation in annotations:
        for j in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[j - 1],
                     annotation[j], (0, 0, 200), 8)

    # Overlay webcam in corner
    imgSmall = cv2.resize(img, (ws, hs))
    imgCurrent[0:hs, width - ws:width] = imgSmall

    # Display windows
    cv2.imshow("Webcam", img)
    cv2.imshow("Presentation", imgCurrent)

    # Exit on 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
