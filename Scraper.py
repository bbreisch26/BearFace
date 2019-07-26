import cv2
import cvutils
import os
counter = 0
imagepath = input('Path of folder holding images with faces (include final slash)')
imglist = os.listdir(imagepath)
print(imglist)

for img in imglist:
    imgstring = imagepath + img
    image = cv2.imread(imgstring)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(125,125), maxSize=(1024,1024))
    if len(faces) == 0:
        # os.remove('C:/Users/Ben/Desktop/Photos/' + images)
        print("[INFO] Found {0} Faces!".format(len(faces)))
    else:
        print("[INFO] Found {0} Faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            if x > 0 or y > 0 or w > 0 or h > 0:
                counter += 1
                imagey = image[y:y+h, x:x+h]
                if image is not None:
                    status = cv2.imwrite(imagepath + str(counter) + '.jpeg', imagey)
                    print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

