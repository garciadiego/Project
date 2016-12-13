#!/usr/bin/python
from PIL import Image
import os, sys

#Image Processing resize to 28x28


path = "/home/diegogarcia/Desktop/QlearningHD/Resize_Image/Normal/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            imResize.save(f + ' resized.png', 'png')

resize()
