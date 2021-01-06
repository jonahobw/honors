from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import sys
import skimage.io
import math
from PIL import Image
from general import *
import pandas as pd

# length of feature vector
FEATURE_LENGTH = 22

def normalize_csv(csv_file, filename = None, save_folder = None):
    # this function will take a csv of raw image feature data and normalize the samples
    # it will create a new csv with the name <csv_file>_normalized.csv (or <filename> if specified)
    # and save it in <save_folder>

    if save_folder == None:
        save_folder = os.path.join(os.getcwd(), "Image_features")
    if filename == None:
        root_folder, csv_name = os.path.split(csv_file)
        csv_name_no_ext = csv_name.split(".")[0]
        new_name = csv_name_no_ext + "_normalized.csv"
        filename = os.path.join(root_folder, new_name)

    # first 2 columns are filename and image class, get all other columns
    columns = [i for i in range(FEATURE_LENGTH+2)]
    columns.remove(0)
    columns.remove(1)
    first_2_cols = pd.read_csv(csv_file, usecols=[0,1])
    img_file_names = first_2_cols[['filename']]
    img_class = first_2_cols[['img_class']]
    df = pd.read_csv(csv_file, usecols = columns)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    normalized_df.insert(loc=0, column = 'img_class', value = img_class)
    normalized_df.insert(loc=0, column = 'filename', value = img_file_names)
    normalized_df.to_csv(filename)
    return filename


def store_csv(test_folder, filename = None, save_folder = None, validation = True):
    # gathers the image feature vector for each image in <test_folder> and stores this data in csv format under a
    # file named <filename> and stored in <save_folder>
    #
    # if <validation> is True, then <test_folder> should be a root directory containing subfolders of "Train"
    # and "Validation" and all of the images in these subfolders will be stored in the csv.  This feature
    # was added because the neural network code partitions the csv into training and validation data

    if save_folder == None:
        save_folder = os.path.join(os.getcwd(), "Image_features")
    if filename == None:
        date = str_date()
        test_folder_abbreviated = os.path.split(test_folder)[1]
        filename = "img_features_" + str(test_folder_abbreviated) + "_" + date + ".csv"

    if validation:
        train_folder = os.path.join(test_folder, "Train")
        val_folder = os.path.join(test_folder, "Validation")
        img_data = gather_test_data(train_folder)
        img_data.extend(gather_test_data(val_folder))
    else:
        img_data = gather_test_data(test_folder)

    csv_filename = os.path.join(save_folder, filename)
    f = open(csv_filename, "w")

    #hard-coded based on the image feature vector
    f.write("filename,img_class,max_color_len,max_color_r,max_color_g,max_color_b,r_avg,g_avg,b_avg,"
            "dom_color1_r,dom_color1_g,dom_color1_b,dom_color2_r,dom_color2_g,dom_color2_b,"
            "dom_color3_r,dom_color3_g,dom_color3_b,dom_color4_r,dom_color4_g,dom_color4_b,"
            "dom_color5_r,dom_color5_g,dom_color5_b,\n")

    #iterate over data
    for file, sign, features in img_data:
        f.write(str(file) + ",")
        f.write(str(sign) + ",")
        for feature in features:
            f.write(str(feature) + ",")
        f.write("\n")

    f.close()


def test_store_csv():
    test_folder = os.path.join(os.getcwd())
    store_csv(test_folder)

def gather_test_data(test_folder):
    # given a folder with subfolders named as the sign classes, this function will traverse through the subfolders
    # and gather all image features of each image, and return an array of 3-tuples of the format:
    # (<file name>, <class name>, <[array of image features]>)
    img_data = []
    classes = os.listdir(test_folder)
    for sign in classes:
        sign_folder = os.path.join(test_folder, sign)
        for file in os.listdir(sign_folder):
            img_file = os.path.join(test_folder, sign, file)
            img_data.append((file, sign, gather_image_features(img_file)))
    return img_data


def gather_image_features(image_file):
    # this function gathers the features of the image from the image_file
    # returns feature vector as an array (currently length of 22)
    # currently does not set optional variables in function calls
    img_features = []

    max_color_len, _, _, _, max_color = get_max_color_length(image_file)
    img_features.append(max_color_len)
    for color_channel in max_color:
        img_features.append(color_channel)

    r, g, b = avg_rgb(image_file)
    img_features.append(r)
    img_features.append(g)
    img_features.append(b)

    dom_colors = get_dominant_colors(image_file)
    for color in dom_colors:
        img_features.extend(list(color))

    return img_features


def get_max_color_length(img_file, threshold = 25, size = (124, 124), color_limits = (0, 0)):
    # finds the longest row or column of pixels in the image whose color differs by no more than the threshold value
    # returns a 5-tuple: <length of the longest row/column>, <start x pixel coordinate>, <start y pixel coordinate>,
    # <direction>, <color>
    #
    # parameters:
    # img_file     (string):    path to an image
    # threshold     (float):    max euclidean distance between 2 colors for them to be considered the same color
    # size  (tuple (int, int)): size to rescale the image to
    # color_limits (tuple(float, float)): color limits[0] is a threshold for how close any color can be to black
    # (all 0's in RGB) and color_limits[1] is a threshold for how close any color can be to white (all 255's in RGB)
    # if a color falls within the threshold, it is not considered for the max color length.  The purpose of
    # color_limits is to filter out the white and black parts of the picture if desired.  It can be turned off if set
    # to (0, 0)

    image = Image.open(img_file)
    image = image.resize(size)
    x_max, y_max = size

    # pixel_directions = dict where each pixel has a set of 4 directions to check, up, down, left, and right.This
    # will look like(x, y): ["up", "down", "left", "right"] where x and y are the pixel coordinates.  Will have to
    # remove some of these directions for pixels on the edges

    pixel_directions = get_pixel_directions(x_max, y_max)

    max_color_length = 0
    # coordinates of the start and end of the longest color length
    x_start = None
    y_start = None
    max_direction = None
    max_color = None

    # iterate through each pixel (x, y)
    for pixel in pixel_directions:
        # iterate through each possible direction, which could be ["up", "down", "left", "right"]
        for direction in pixel_directions[pixel]:
            current_x = pixel[0]
            current_y = pixel[1]

            if distance_to_edge(current_x, current_y, x_max, y_max, direction) <= max_color_length:
                # do not check this pixel direction because it cannot be greater than max_color_length
                # could also remove all pixels from this pixel to the edge, but this "if" statement will be triggered by all of those pixels as well
                # so it is not necessary
                continue

            reference_color = image.getpixel((current_x, current_y))  # this is an array of length 3 [r, g, b]

            # check if color is too close to black or white
            black_threshold, white_threshold = color_limits
            if (within_threshold([0,0,0], reference_color, black_threshold) or
                    within_threshold([255,255,255], reference_color, white_threshold)):
                continue

            current_color_length = 1

            # get next pixel and color
            current_x, current_y = get_next_pixel(current_x, current_y, x_max, y_max, direction)
            outside_image = current_x == -1
            current_color = image.getpixel((current_x, current_y))

            while (within_threshold(reference_color, current_color, threshold) and not outside_image):
                current_color_length += 1
                # remove opposite direction from current pixel
                if opposite_direction(direction) in pixel_directions[(current_x, current_y)]:
                    pixel_directions[(current_x, current_y)].remove(opposite_direction(direction))

                current_x, current_y = get_next_pixel(current_x, current_y, x_max, y_max, direction)
                outside_image = current_x == -1
                if outside_image:
                    break
                current_color = image.getpixel((current_x, current_y))

            # see if this is a new max
            if (current_color_length > max_color_length):
                max_color_length = current_color_length
                x_start, y_start = pixel
                max_direction = direction
                max_color = reference_color

    return max_color_length, x_start, y_start, max_direction, max_color

def get_pixel_directions(x_max, y_max):
    pixel_directions = {}
    for y in range(x_max):
        for x in range(y_max):
            directions = []
            if x!=0:
                directions.append("left")
            if x!=x_max:
                directions.append("right")
            if y!=0:
                directions.append("up")
            if y!=y_max:
                directions.append("down")
            pixel_directions[(x, y)] = directions
    return pixel_directions

def get_next_pixel(x, y, x_max, y_max, direction):
    # gives x and y coordinates of the next pixel starting from x, y going in <direction>
    # if the next pixel would be outside the image size (x_max, y_max), the function returns (-1, -1)
    # starting coordinate of (0, 0) is the top left pixel of the image, so moving down the image increases y

    if direction == "up":
        if y - 1 < 0:
            return (-1, -1)
        else:
            return (x, y - 1)
    if direction == "down":
        if y + 1 >= y_max:
            return (-1, -1)
        else:
            return (x, y + 1)
    if direction == "right":
        if x + 1 >= x_max:
            return (-1, -1)
        else:
            return (x + 1, y)
    if direction == "left":
        if x - 1 < 0:
            return (-1, -1)
        else:
            return (x - 1, y)
    else:
        return (-1, -1)

def within_threshold(color1, color2, threshold):
    # color1 and color2 are 1d arrays of length 3 representing a color [r, g, b]
    # function computes the euclidean distance between these two colors and returns a boolean indicating if they are within the threshold distance of
    # each other
    distance = 0
    for i in range(3):
        distance += (color1[i] - color2[i])**2
    distance = math.sqrt(distance)
    if distance < threshold:
        return True
    else:
        return False

def distance_to_edge(x, y, x_max, y_max, direction):
    if direction == "up":
        return y
    if direction == "down":
        return y_max - y
    if direction == "left":
        return x
    if direction == "right":
        return x_max - x

def opposite_direction(direction):
    if direction == "up":
        return "down"
    if direction == "down":
        return "up"
    if direction == "left":
        return "right"
    if direction == "right":
        return "left"

def avg_rgb(img_file, plot = False):
    # adapted from https://datacarpentry.org/image-processing/aio/index.html
    # returns a 3-tuple: (red average, green average, blue average)
    image = skimage.io.imread(fname=img_file)

    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    avgs = {}
    r = list(np.concatenate(image[:, :, 0]))
    b = list(np.concatenate(image[:, :, 1]))
    g = list(np.concatenate(image[:, :, 2]))
    avgs["r"] = sum(r) / len(r)
    avgs["g"] = sum(g) / len(g)
    avgs["b"] = sum(b) / len(b)

    if plot:
        # create the histogram plot, with three lines, one for
        # each color
        plt.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=c)

        plt.legend()
        plt.xlabel("Color value")
        plt.ylabel("Pixels")

        plt.show()
    return avgs["r"], avgs["g"], avgs["b"]


def get_dominant_colors(image_file, number_of_colors = 5, show_chart = False):
    # adapted from https://github.com/kb22/Color-Identification-using-Machine-Learning/blob/master/Color%20Identification%20using%20Machine%20Learning.ipynb
    # gets top <number_of_colors> colors from an image using K means clustering

    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    modified_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.pie(counts.values(), colors=hex_colors)
        ax2 = fig.add_subplot(122)
        ax2.imshow(image)
        ax2.axis('off')
        plt.show()

    return rgb_colors

#file = os.path.join(os.getcwd(), 'Debug', '00', '03922.png')
file = os.path.join(os.getcwd(), 'Debug', '00', '06946.png')
print(get_max_color_length(file))
#print(get_dominant_colors(file, 5, True))
#print(gather_image_features(file))
#test_folder = os.path.join(os.getcwd(), "small_test_dataset", "Train")
#print(gather_test_data(test_folder))
#test_store_csv()
# csv = os.path.join(os.getcwd(), "Image_features", "img_features_GTSRB_ResNet_2020-12-29.csv")
# a = normalize_csv(csv)
# print(a)