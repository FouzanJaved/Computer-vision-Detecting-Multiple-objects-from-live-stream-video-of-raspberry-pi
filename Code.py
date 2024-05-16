# Here is the code to recieve the video. As the video is send in the format of MPEG so it should be recieved in the same pattern:-

import cv2
import urllib 
import numpy as np

stream = urllib.urlopen('http://192.168.1.20:8081/frame.mjpg')
bytes = ''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),  cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)   


#Server side

import cv2
import urllib 
import numpy as np
MIN_MATCH_COUNT=30

detector=cv2.SIFT()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("training/1.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

trainImg2=cv2.imread("training/2.jpg",0)
trainKP2,trainDesc2=detector.detectAndCompute(trainImg2,None)


stream = urllib.urlopen('http://192.168.1.18:8081/frame.mjpg')
bytes = ''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),  cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        
        #ret, QueryImgBGR=cam.read()
        QueryImg=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        
        queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
        matches=flann.knnMatch(queryDesc,trainDesc,k=2)

        queryKP2,queryDesc2=detector.detectAndCompute(QueryImg,None)
        matches2=flann.knnMatch(queryDesc2,trainDesc2,k=2)

        goodMatch=[]
        goodMatch2=[]
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)
        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp=[]
            qp=[]
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp=np.float32((tp,qp))
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w=trainImg.shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(i,[np.int32(queryBorder)],True,(0,255,0),5)

            cv2.putText(i,'Sign detected!', 
            (0,300), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,0,0),
            2)
            
            cv2.putText(i,'Maintain constant vehicle speed', 
            (0,350), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,0,0),
            2)

            
        else:
            print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        cv2.imshow('result',i)


        for m,n in matches2:
            if(m.distance<0.75*n.distance):
                goodMatch2.append(m)
        if(len(goodMatch2)>MIN_MATCH_COUNT):
            tp2=[]
            qp2=[]
            for m in goodMatch2:
                tp2.append(trainKP2[m.trainIdx].pt)
                qp2.append(queryKP2[m.queryIdx].pt)
            tp2,qp2=np.float32((tp2,qp2))
            H,status=cv2.findHomography(tp2,qp2,cv2.RANSAC,3.0)
            h,w=trainImg.shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(i,[np.int32(queryBorder)],True,(0,255,0),5)

            cv2.putText(i,'Sign detected 2!', 
            (0,300), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,0,0),
            2)
            
            cv2.putText(i,' ', 
            (0,350), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,0,0),
            2)

            
        else:
            print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        cv2.imshow('result',i)


        if cv2.waitKey(10)==ord('q'):
            break
cam.release()
cv2.destroyAllWindows()   