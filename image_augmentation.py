import cv2
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte, exposure
import os
from skimage.util import random_noise
import random

# adapted from https://github.com/govinda007/Images/blob/master/augmentation.ipynb

def load_im_cv2(imfile, size = 124):
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))

def im_exposure(im, r = None):
    if r == None:
        r = random.random()*2.8+0.3
    return exposure.adjust_gamma(im, r)

def show_im(im):
    plt.imshow(im)
    plt.show()

def crop_im(im, p = 0.3):
    h, w, _ = dimensions(im)
    startx = int(random.random()*p * h)
    endx = int((random.random()+(1-p)) * h)
    starty = int(random.random()*p*w)
    endy = int((random.random()+(1-p))*w)
    return im[startx:endx, starty:endy]

def dimensions(im):
    height, width, dims = im.shape
    return height, width, dims

def rotate_im(im, angle=25, rand=True):
    if rand:
        range = 2*angle
        angle = random.random()*range-angle
    return rotate(im, angle=angle)

def add_noise(im):
    return random_noise(im)

def blur_im(im, blur=21, rand=True):
    # expects an image of size 124 x 124
    if rand:
        blur = int(random.random()*blur)
        while(blur%2 != 1):
            blur += 1
    return cv2.GaussianBlur(im, (blur, blur), 0)

def generate_imgs(original_path, new_path, number_to_generate, verbose = False):
    # original path  - path to original images
    # new path - where to save new images
    # number_to_generate - how many new images to make

    transformations = {'rotate': rotate_im,
                       'noise': add_noise,
                       'blur': blur_im,
                       'exposure': im_exposure,
                       'crop': crop_im
                       }  # use dictionary to store names of functions

    images = []  # to store paths of images from folder

    for im in os.listdir(original_path):  # read image name from folder and append its path into "images" array
        images.append(os.path.join(original_path, im))

    i = 1  # variable to iterate till images_to_generate

    while i <= number_to_generate:
        if verbose and i%100 ==0:
            print("{} images generated".format(str(i)))
        image = random.choice(images)
        original_image = load_im_cv2(image)
        transformed_image = None
        n = 0  # variable to iterate till number of transformation to apply
        transformation_count = random.randint(1, len(
            transformations))  # choose random number of transformation to apply on the image

        while n <= transformation_count:
            key = random.choice(list(transformations))  # randomly choosing method to call
            transformed_image = transformations[key](original_image)
            n = n + 1
        new_image_path = os.path.join(new_path, "augmented_" + str(i) + ".png")

        transformed_image = img_as_ubyte(
            transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)  # convert image to RGB before saving it
        cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
        i = i + 1

def augment_dataset(train_folder, num_imgs, verbose = True):
    # expects train_folder to have subfolders with the name of the classes
    # num_imgs is the minimum number of images (per class) that the augmented dataset will have

    classes = os.listdir(train_folder)

    for class_name in classes:
        class_folder = os.path.join(train_folder, class_name)
        existing_imgs = len(os.listdir(class_folder))
        imgs_to_generate = num_imgs - existing_imgs
        if verbose:
            print("\nWorking on class {}, generating {} images".format(class_name, str(imgs_to_generate)))
        if imgs_to_generate > 0:
            generate_imgs(class_folder, class_folder, imgs_to_generate, verbose = verbose)

    return


if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted", "white_circular_fc_augmented", "Train")
    augment_dataset(folder, 2000)

    # folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted", "circle_color", "Debug")
    # augment_dataset(folder, 10)

    # img = os.path.join(os.getcwd(), "Debug", "00", "03922.png")
    # im = load_im_cv2(img)
    # im = crop_im(im)
    # show_im(im)