import cv2
import numpy as np


try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

img = cv2.imread('license-recognition/data/rabo-license-dataset/scaled/budi.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# If tesseract is not in PATH, include this:
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# List of available languages
#print(pytesseract.get_languages(config=''))

# Adding custom options
#custom_config = r'--oem 3 --psm 6'
print(pytesseract.image_to_string(gray))


# load the input image
# Converting the image into grayscale

cascade_faces = cv2.CascadeClassifier('license-recognition/models/haarcascade_frontalface_default.xml')
faces = cascade_faces.detectMultiScale(gray, 1.1, 4)
# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# #
# # # Display the output
cv2.imshow('img', img)
cv2.waitKey()
