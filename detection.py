import cv2
import sys
from keras.models import load_model


model = load_model('my_model.h5')

import numpy as np

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')  


# Initialize object of EMR class

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX




while True:
    # Again find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        img = cv2.resize(roi_gray,(48,48))
        img = np.reshape(img,[1,1,48,48])
        classes = model.predict(img)

    	maxindex = np.argmax(classes[0])
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	cv2.putText(frame,EMOTIONS[maxindex],(10,360), font, 2,(255,255,255),2) 



    cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()