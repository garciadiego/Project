
from scipy import misc
import PIL
from PIL import Image
import numpy
import os, sys


#to display the shape of the image

#a = misc.imread('/home/diegogarcia/Desktop/QlearningHD/Resize_Image/BW/1.png')
#print a.shape # (300,400,3) means 400x300 with RGB channels
#print a.dtype # 'uint8'

#Image Processing from RGB to Gray

path = "/home/diegogarcia/Desktop/QlearningHD/Resize_Image/Resized_img/"
dirs = os.listdir( path )


def resize():
    for item in dirs:
        im = misc.imread(path+item)
        f, e = os.path.splitext(path+item)
        #imResize = im.resize((28,28), Image.ANTIALIAS)
        imResize = im[:, :, 0]
        print imResize.shape
        Image.fromarray(imResize).save(f + ' b&w.png', 'png')
resize()
