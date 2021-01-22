import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from pytorch_resnet import test_one_image
from general import *
import random
import logging
from tree_helper import attack_danger_weights
from nndt import nndt_depth3_unweighted

logger = logging.getLogger("attack_log")

def print_image(img, path = True, title = ""):
    # displays an image with title = title
    if path:
        img = mpimg.imread(img)
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def spans_multiple_classifiers(array, class1, class2):
    # <array>: a 2d array where the first dimension are the leaf classifiers of the nndt
    # and the second dimension are the classes classified by the leaf classifiers.
    # class1 and class2 are ints representing two signs
    # returns a boolean indicating !(class1 and class2 are in the same 2nd-dimension array)

    for subarr in array:
        # variables indicating if the classes have been found
        cl1 = class1 in subarr
        cl2 = class2 in subarr
        if cl1 and cl2:
            # 2 signs are in the same leaf classifier, they do NOT span multiple classifiers
            return False
        elif cl1 or cl2:
            return True



def get_correctly_classified_imgs(model, imgs, samples, nndt = False):
    # This function takes an array of image files and returns a subset of that array of size <samples> where
    # every image in the subset is correctly classified by the <model>
    #
    # parameters:
    # model (neural network object or nndt): the model to use for classification
    # nndt (bool) indicates whether or not the model is an nndt
    # imgs (list): array of tuples (image file name, image true class)
    # samples (int): the number of images desired who classify correctly

    valid_imgs = []
    for i in range(len(imgs)):
        image, true_class = imgs.pop()
        if not nndt:
            im = Image.open(image)
            preds = test_one_image(model, im)
        else:
            preds = model.prediction_vector(image, dict=False)
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


def retrieve_valid_test_images(model, image_folder, samples, targeted = None, exclusive = None, nndt = False):
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
        return get_correctly_classified_imgs(model, imgs, samples, nndt=nndt)

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
    return get_correctly_classified_imgs(model, imgs, samples, nndt=nndt)


def retrieve_attack_pairs(n, rand = False, across_classifiers = None):
    # if rand = False: gets n highest attack pairs by weight and returns an array of tuples
    # (startsign, endsign) in descending order of attack weight
    #
    # if rand = True, same as above except that the attack pairs are chosen randomly
    # if across_classifiers is not none, it is a 2d array where the 1st dimension are final classifiers in an nndt
    # and the second dimension is the signs in each final classifier.  If across_classifiers is not none, the function
    # will return only attack pairs that span multiple final classifiers.
    attack_pairs = []

    for startsign in range(43):
        for endsign in range(43):
            if startsign != endsign:
                if across_classifiers is not None:
                    if spans_multiple_classifiers(across_classifiers, startsign, endsign):
                        attack_pairs.append((startsign, endsign, attack_danger_weights(startsign, endsign)))
                    continue
                attack_pairs.append((startsign, endsign, attack_danger_weights(startsign, endsign)))

    #random case
    if rand:
        random.shuffle(attack_pairs)
        attack_pairs = attack_pairs[:n]
        for i in range(n):
            st, end, weight = attack_pairs[i]
            attack_pairs[i] = (st, end)
        return attack_pairs

    attack_pairs.sort(key=lambda x: x[2], reverse = True)
    attack_pairs = attack_pairs[:n]

    for i in range(n):
        st, end, weight = attack_pairs[i]
        attack_pairs[i] = (st, end)
    return attack_pairs