import numpy as np
import cv2
import HandTrackingModule as htm
import time
import autopy

###########################

wCam, hCam = 640, 480
frameR = 100
smoothening = 20
pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0


###########################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScreen, hScreen = autopy.screen.size()
# print(wScreen,hScreen)


while True:
    success, img = cap.read()
    cv2.flip(img,1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x1, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR),
                      (wCam-frameR, hCam-frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            try:
                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))
                cLocX = pLocX + (x3 - pLocX)/smoothening
                cLocY = pLocY + (y3 - pLocY)/smoothening

                autopy.mouse.move(wScreen-cLocX, cLocY)
                pLocX, pLocY = cLocX, cLocY

            except:
                pass
            x1, y1 = lmList[8][1:]
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        if fingers[1] == 1 and fingers[2] == 1:
            lenght, img, lineInfo = detector.findDistance(8, 12, img)
            print(lenght)
            if lenght < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
