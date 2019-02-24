import cv2 as cv
import numpy as np
from time import time
from keras.models import load_model
from skimage.transform import resize

cap = cv.VideoCapture(0)
casc= cv.CascadeClassifier('Mouth.xml')
f=0
print('running')
model=load_model('logs')#placeholder
font = cv.FONT_HERSHEY_SIMPLEX
buffer=[]
x,y,w,h=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if f%5==0:
#     # Our operations on the frame come here
    if f%5==0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        # cv.imshow('frame',frame)
        mouth = casc.detectMultiScale(gray, 1.1, 100)
        x=mouth[0][1]
        y=mouth[0][0]
        w=mouth[0][3]
        h=mouth[0][2]
    img=frame[x-60:x+w,y-10:y+h+10,:]
    buffer.append(frame)
    if len(buffer>6):
        buffer.remove(buffer[0])
    if f%3==0 and f>=6:
        if model.predict(buffer)[0]>.50:
            cv.putText(frame,'BAD WORD', (100,100),font, 7,(255,0,0),2,cv.LINE_AA)


            # print(x,' ',y,' ',w,' ',h)
        # print(mouth[0])
    f+=1

    cv.imshow('img',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
