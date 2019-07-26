from PIL import Image
import os
path = input('Path of folder holding photos')
filelist = os.listdir(path)

for files in filelist:
    img = Image.open(path + files)
    img2 = img.resize((160, 160))
    os.remove(path + files)
    img2.save(path + files)
