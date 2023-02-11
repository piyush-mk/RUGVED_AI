import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

import time

    
    
cap = cv2.VideoCapture(0)
redLower = np.array([110,50,60], dtype='uint8')
redUpper = np.array([140,255,255], dtype='uint8')
c = 0
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('../vid.mp4',cv2.VideoWriter_fourcc(*'MPEG'),10, size) 
while True:
    grapped,frame=cap.read()
    if grapped == True:
        
        red = cv2.inRange(frame,redLower,redUpper)
        red = cv2.GaussianBlur(red,(3,3),0)

        cnts = cv2.findContours(red.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.circle(frame, (rect[0][0]+(rect[-1][0] - rect[0][0])//2,rect[1][1]+(rect[-1][-1]-rect[1][1])//2), 40, (0, 0, 0), 1)
            
        #cv2.imshow("Ball Tracking", frame)
        result.write(frame)
        #if cv2.waitKey() & 0xFF == ord("q"):
         #   break

    else:
        break
        
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
cap.release()
cv2.destroyAllWindows()
