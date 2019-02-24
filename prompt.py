import numpy as np
import cv2 as cv
from time import time
import glob
import os
from skimage.transform import resize

font = cv.FONT_HERSHEY_SIMPLEX
f=0
options=(['horse','dog','fire','barn','fish','run','hi','weather',"I'm",'yours','Mine',
    'breath','lead','metal','owl','brain','storm','tick','snake','this','is','ice','pick',
    'correct','who','what','where','hill','camp','clam','calm','drop']) #various filler words


def select(num):#will select word next to be recorded
    if num%2==0:
        return 'crap'
    else:
        return options[np.random.randint(0,len(options))]
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def cleanPhotos():
    casc= cv.CascadeClassifier('Mouth.xml') #makes everything work
    start=int(time())#helps group images
    filenames = glob.glob("data/posBase/*.jpg")# grab all the pphotos into a list
    images = [cv.imread(img) for img in filenames]
    e=0
    for image in images: #crops every image currently uncropped
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mouth = casc.detectMultiScale(gray, 1.1, 100)
        print(f'cleaning {filenames[e]} {round(e/len(filenames)*100, 1)}% complete')
        x=mouth[0][1]
        y=mouth[0][0]
        w=mouth[0][3]
        h=mouth[0][2]
        img=image[x-60:x+w,y-10:y+h+10,:] # really likes focusing on my chin so this makes it larger
        img=resize(img,[140,140,3],preserve_range=True) #this is actually larger than standard
        cv.imwrite(f'data/pos/{start}{filenames[e][13:]}',img)#save photo with unique name
        try:
            os.remove(filenames[e])# delete old photo from staging folder
        except: pass
        e+=1
    print('negative images')# halfway there
    filenames = glob.glob("data/negBase/*.jpg")
    images = [cv.imread(img) for img in filenames]
    e=0
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mouth = casc.detectMultiScale(gray, 1.1, 100)
        print(f'cleaning {filenames[e]} {round(e/len(filenames)*100, 1)}% complete')
        x=mouth[0][1]
        y=mouth[0][0]
        w=mouth[0][3]
        h=mouth[0][2]
        img=image[x-60:x+w,y-10:y+h+10,:]
        img=resize(img,[140,140,3],preserve_range=True)
        cv.imwrite(f'data/neg/{start}{filenames[e][13:]}',img)
        try:
            os.remove(filenames[e])
        except: pass
        e+=1

def impute():# this may be used to artificially increase the amount of images avaliable
    return null

def go():#this function gathers the photos
    cap = cv.VideoCapture(0)#grab from webcam
    word=0
    curr=select(word)
    buffer=[]
    f=0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame=cv.flip(frame, 1)
        buffer.append(frame)
        if len(buffer)>15:
            buffer.remove(buffer[0])
        cv.putText(frame, f'{curr} {word}', (100,100),font, 4,(0,0,0),2,cv.LINE_AA)#shows which image is
        if cv.waitKey(1) & 0xFF == ord('d'):
            word+=1
            e=1
            if word%2!=0:
                type='pos'
            else:
                type='neg'
            for i in range(3,len(buffer)-6):
                cv.imwrite(f'data/{type}Base/{curr}_{word}-{e}.jpg', buffer[i])
                e+=1
            buffer=[]
            if word==200:
                break
            curr=select(word)

        cv.imshow('img',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print('Thanks!')
    cleanPhotos()
if __name__=='__main__':
    print('Updated')
