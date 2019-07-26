import cv2
import os
import re


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


filepath = input('Enter path of files that you would like to be numbered (include final slash)')
imglist = os.listdir(filepath)
counter = 1
imglist = sorted_aphanumeric(imglist)
print(imglist)
for imagename in imglist:
    os.rename(filepath + imagename, filepath + str(counter) + '.jpg')
    counter = counter + 1
