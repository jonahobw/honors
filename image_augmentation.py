import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
import random

def load_im_cv2(imfile, size = 124):
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))

def show_im(im):
    plt.imshow(im)
    plt.show()

def rotate_im(im, angle=15, rand=True):
    if rand:
        range = 2*angle
        angle = random.random()*range-angle
    return rotate(im, angle=angle)

def add_noise(im):
    return random_noise(im)

def blur_im(im, blur=21, rand=True):
    # expects an image of size 124 x 124
    if rand:
        blur = random.random()*blur
        while(blur%2 != 1):
            blur +=1
    return cv2.GaussianBlur(im, (blur, blur), 0)