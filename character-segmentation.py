import sys
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt

#image File
im = cv2.imread('/home/diegogarcia/Desktop/QlearningHD/segmen/data/equ.jpg')
im3 = im.copy()

#image file is scan for symbols and saved on an array
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#Print Image Contours
plt.subplot(121),plt.imshow(gray),plt.title('RGB to Gray Scale')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.show()



samples =  np.empty((0,100))
detection = []
keys = [i for i in range(48,58)] #press spacebar for one by one scan (maybe a loop later ?)

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27: #esc 
                sys.exit()
            elif key in keys:
                detection.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)


detection = np.array(detection,np.float32)
detection = detection.reshape((detection.size,1))

#output File
pickle.dump(samples, open(‘/FolderLocation’,’wb'))
pickle.dump(detection, open(‘/FolderLocation’,’wb'))


print "Scanning Complete"
