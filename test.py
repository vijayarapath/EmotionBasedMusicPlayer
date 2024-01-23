#!/usr/bin/env python
# coding: utf-8

# In[1]:

import cv2
import pygame

# import IPython

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from deepface import DeepFace

# In[4]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[5]:


cap=cv2.VideoCapture(0)


# In[8]:


while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if result:
        cv2.putText(frame,
           result['dominant_emotion'],
           (50,50),
           font,3,
           (255,0,0),
           2,
           cv2.LINE_4)
        cv2.imshow('Video',frame)
        if(result['dominant_emotion'] == 'happy'):
           pygame.mixer.init()
           pygame.mixer.music.load("song1.mp3")
           pygame.mixer.music.play()

        elif(result['dominant_emotion'] == 'sad'):
           pygame.mixer.init()
           pygame.mixer.music.load("song.mp3")
           pygame.mixer.music.play()

        elif(result['dominant_emotion'] == 'neutral'):
           pygame.mixer.init()
           pygame.mixer.music.load("song.mp3")
           pygame.mixer.music.play()

        elif(result['dominant_emotion'] == 'angry'):
           pygame.mixer.init()
           pygame.mixer.music.load("song.mp3")
           pygame.mixer.music.play()

    else:
        pass
        #raise IOError("Face not Detected..")
        
    if cv2.waitKey(2) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[64]:


if not cap.isOpened():
    raise IOError("Cannot open webcam")


# In[10]:


print(result)


# In[ ]:


print(result)
# IPython.display.display(IPython.display.Audio("song.mp3"))

# In[ ]:




