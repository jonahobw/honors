import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from pytorch_resnet import test_one_image
from general import *
import random
import logging
from tree_helper import attack_danger_weights

logger = logging.getLogger("attack_log")

def print_image(img, path = True, title = ""):
    # displays an image with title = title
    if path:
        img = mpimg.imread(img)
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def get_correctly_classified_imgs(model, imgs, samples):
    # This function takes an array of image files and returns a subset of that array of size <samples> where
    # every image in the subset is correctly classified by the <model>
    #
    # parameters:
    # model (neural network object): the model to use for classification
    # imgs (list): array of tuples (image file name, image true class)
    # samples (int): the number of images desired who classify correctly

    valid_imgs = []
    for i in range(len(imgs)):
        image, true_class = imgs.pop()
        im = Image.open(image)
        preds = test_one_image(model, im)
        class_pred = preds.index(max(preds))
        if(int(class_pred)==int(true_class)):
            valid_imgs.append((image, true_class))
            samples -= 1
        if(samples == 0):
            break

    if samples > 0:
        logger.error("Error, unable to find {} correctly classified images from provided data".format(str(samples)))
        return -1
    else:
        return valid_imgs


def retrieve_valid_test_images(model, image_folder, samples, targeted = None, exclusive = None):
    # returns a set of images who classify correctly of size <samples>
    #
    # if <targeted> is not none, it should be an integer representing one of the classes, and
    # the function will exclude images with a true class of <targeted>
    #
    # if <exclusive> is not none, it should be an integer representing one of the classes, and
    # the function will only include images with a true class of <exclusive>

    imgs = []

    # <exclusive> case:
    if (exclusive is not None):
        sign_folder = os.path.join(image_folder, format_two_digits(exclusive))
        test_data = os.listdir(sign_folder)
        for i in range(len(test_data)):
            file = test_data[i]
            file_path = os.path.join(sign_folder, file)
            # tuple of (image filename, label of image)
            imgs.append((file_path, exclusive))
        return get_correctly_classified_imgs(model, imgs, samples)

    classes = os.listdir(image_folder)

    for sign in classes:
        if(targeted is not None and int(targeted)==int(sign)):
            continue
        sign_directory = os.path.join(image_folder, str(sign))
        test_data = os.listdir(sign_directory)
        for i in range(len(test_data)):
            file = test_data[i]
            file_path = os.path.join(sign_directory, file)
            # tuple of (image filename, label of image)
            imgs.append((file_path, sign))
    random.shuffle(imgs)
    return get_correctly_classified_imgs(model, imgs, samples)


def highest_attack_pairs(n):
    # gets n highest attack pairs by weight and returns an array of tuples
    # (startsign, endsign) in descending order of attack weight
    attack_pairs = []

    for startsign in range(43):
        for endsign in range(43):
            attack_pairs.append((startsign, endsign, attack_danger_weights(startsign, endsign)))

    attack_pairs.sort(key=lambda x: x[2], reverse = True)
    attack_pairs = attack_pairs[:n]

    for i in range(n):
        st, end, weight = attack_pairs[i]
        attack_pairs[i] = (st, end)
    return attack_pairs