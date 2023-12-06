import cv2
import cvzone as cvz
import numpy as np

cap = cv2.VideoCapture(0)

#cap.set(3,640)
#cap.set(4,380)

def preProc(img):

    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre, 219,240)

    # to close the perimeter and stuff
    kernel = np.ones((3,3),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

while True:
    success, img = cap.read()
    width = int(cap.get(3))  # give the width property of the frame --- 3 = id of property of vd cap
    height = int(cap.get(4))

    imgPre = preProc(img)

    imgContours, conFound = cvz.findContours(img,imgPre,minArea=20)

    totalMoney =0

    #imgCount = np.zeros((480, 640, 3), np.uint8)

    if conFound :
        for cont in conFound :
            peri = cv2.arcLength(cont['cnt'],True)
            approx = cv2.approxPolyDP(cont['cnt'],0.02*peri,True)

            if len(approx)>5:
                #print(cont['area'])
                area = cont['area']

                if area < 500 :
                    totalMoney += 0.5
                elif 2500 < area < 2800 :
                    totalMoney += 1
                elif 2900 < area < 3020 :
                    totalMoney += 5
                elif 3030 < area < 3200 :
                    totalMoney += 2
                else :
                    totalMoney += 10

   # print(totalMoney)

    #cvz.putTextRect(imgCount,f'{totalMoney} Dhs', (100,200), 10,30,7)
    #imgStacked = cvz.stackImages([img,imgPre,imgContours,imgCount],2,1)
    
    imgStacked = cvz.stackImages([img, imgPre, imgContours], 2, 1)
    cvz.putTextRect(imgStacked, f'{totalMoney} Dhs', (50,50 ))
    cv2.imshow('Frame', imgStacked)

    if cv2.waitKey(1)== ord('q'):  # return the ascii value of the key we press to close the window
        break