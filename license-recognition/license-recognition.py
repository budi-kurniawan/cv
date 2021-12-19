import cv2

import numpy as np
# load the input image
#path = 'c:/users/budik/PycharmProjects/pythonProject1/test.jpeg'
img = cv2.imread('rabo-license-dataset/scaled/10.jpg')

# Converting the image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loading the cascade
cascade_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # Detecting faces
faces = cascade_faces.detectMultiScale(gray, 1.1, 4)
#
# # Drawing rectangle around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
# # Display the output
cv2.imshow('img', img)
cv2.waitKey()
