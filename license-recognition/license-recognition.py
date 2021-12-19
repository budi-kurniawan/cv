import cv2
import numpy as np


try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def resize(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def main():
    input_path = 'license-recognition/data/rabo-license-dataset/'
    output_path = 'license-recognition/output/'
    cascade_faces = cv2.CascadeClassifier('license-recognition/models/haarcascade_frontalface_default.xml')
    for i in range(1, 51):
        img = cv2.imread(input_path + str(i) + '.jpg')
        img = resize(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_faces.detectMultiScale(gray, 1.1, 4)
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(output_path + str(i) + '.jpg', img)
        text = pytesseract.image_to_string(gray)
        with open(output_path + str(i) + '.txt', 'w') as f:
            f.write(text)

if __name__ == '__main__':
    main()