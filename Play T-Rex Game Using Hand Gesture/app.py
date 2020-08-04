import  numpy as np
import  cv2 as cv
import  pyautogui
import  math

cap = cv.VideoCapture(0)
while True :
    ret , frame = cap.read()

    #collect Hand Gesture
    cv.rectangle(frame,(100,100),(300,300),(255,0,0),1,cv.LINE_AA)
    hand_img = frame[100:300 , 100:300]

    #blur the image
    blur = cv.GaussianBlur(hand_img,(3,3),0)

    #convert the RGB colored image to HSV colored image
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)

    #binary image transformation where skin will be white in color and rest will be black in color
    skin = cv.inRange(hsv,np.array([2,0,0]),np.array([25,255,255]))

    #Applying dilation to reduce unwanted black dots on the image
    kernel = np.ones((2,2))
    dilation = cv.dilate(skin,kernel,iterations=2)

    #Appplying Erosion on the image to erode extra white color to be in shape the image
    erosion = cv.erode(dilation,kernel,iterations=2)

    filtered = cv.GaussianBlur(erosion,(3,3),0)
    ret , thresh = cv.threshold(filtered,127,255,0)

    #finding contours
    contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv.contourArea(x))

        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(hand_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        hull = cv.convexHull(contour)

        draw = np.zeros(hand_img.shape, np.uint8)
        cv.drawContours(draw, [contour], -1, (0, 255, 0), 0)
        cv.drawContours(draw, [hull], -1, (0, 0, 255), 0)

        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 90:
                count_defects += 1
                cv.circle(hand_img, far, 1, [0, 0, 255], -1)

            cv.line(hand_img, start, end, [0, 255, 0], 2)

        # if the codition matches, press space
        if count_defects >= 4:
            pyautogui.press('space')
            cv.putText(frame, "JUMP", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except:
        pass

    cv.imshow("Gesture", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()