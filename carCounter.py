import cv2
import numpy as np

cap = cv2.VideoCapture("OpenCV/Storage/CarVideo.mp4")
wMin, hmin = 80,80
posLine= 221
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def centerFunc(x,y,w,h):
    x1 = int(w/2)
    y2 = int(h/2)
    cx = x1+x
    cy = y2+y
    return cx,cy

detect = []
count = 0

while True:
    success, vid = cap.read()
    Vgray = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    Vblur = cv2.GaussianBlur(Vgray,(3,3),5)
    img_sub = algo.apply(Vblur)
    Vdilate = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(Vdilate, cv2.MORPH_CLOSE,kernel)
    counter,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (m,n) in enumerate(counter):
        (x,y,w,h) = cv2.boundingRect(n)
        valCounter = (w>=wMin) and (h>=hmin)
        if not valCounter:
            continue

        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),2)
        #make a circle
        center = centerFunc(x,y,w,h)
        detect.append(center)
        cv2.circle(vid,center,4,(0,0,255),-1)
        #Detection
        for (x,y) in detect:
            if x<posLine and y<posLine:
                count+=1
            detect.remove((x,y))
        
    cv2.putText(vid,"Priyova",(900,100),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),5)
    cv2.putText(vid,str(count),(1700,100),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),5)
    cv2.namedWindow("Car Counter", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Mask", mask)
    cv2.imshow("Car Counter", vid)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break