from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from skimage.feature import hog
from skimage import data, exposure
import os
import sys
import skimage.io
import math
from PIL import Image
from general import *
import pandas as pd
from tree_helper import split_all

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


def store_csv_hog(bins = 12, img_folder = None, filename = None, save_folder = None, validation = True):
    # gathers the HOG for each image in <img_folder> and stores this data in csv format under a
    # file named <filename> and stored in <save_folder>
    #
    # if <validation> is True, then <img_folder> should be a root directory containing subfolders of "Train"
    # and "Validation" and all of the images in these subfolders will be stored in the csv.  This feature
    # was added because the neural network code partitions the csv into training and validation data

    if img_folder == None:
        img_folder = os.getcwd()
    if filename == None:
        date = str_date()
        test_folder_abbreviated = os.path.split(img_folder)[1]
        filename = str(bins) + "hog_img_features_" + str(test_folder_abbreviated) + "_" + date + ".csv"
    if save_folder == None:
        root_folder = filename.split(".")[0]
        save_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    if validation:
        train_folder = os.path.join(img_folder, "Train")
        val_folder = os.path.join(img_folder, "Validation")
        img_data = gather_hog(train_folder, bins)
        img_data.extend(gather_hog(val_folder, bins))
    else:
        img_data = gather_hog(img_folder, bins)

    csv_filename = os.path.join(save_folder, filename)
    f = open(csv_filename, "w")

    f.write("filename,img_class")

    for i in range(bins):
        f.write(",bin " + str(i))
    f.write("\n")

    #iterate over data
    for file, sign, hog in img_data:
        f.write(str(file) + ",")
        f.write(str(sign))
        for feature in hog:
            f.write("," + str(feature))
        f.write("\n")

    f.close()


def load_hog_df(filename, usecols = None):
    root_folder = filename.split(".")[0]
    filename = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, filename)
    if usecols is not None:
        df = pd.read_csv(filename, usecols = usecols)
        return df
    else:
        df = pd.read_csv(filename)
        return df


def kmeans_hog(filename, k):
    # get number of bins from filename
    bins = ""
    for i in filename:
        if i.isdigit():
            bins += i
        else:
            break
    bins = int(bins)
    cols = [x for x in range(2, bins+2, 1)]
    df = load_hog_df(filename, usecols=cols)
    a = df.values
    kmeans = KMeans(n_clusters=k)
    preds = kmeans.fit_predict(a)
    labels = kmeans.labels_
    sil = silhouette_score(a, labels, metric='euclidean')
    # write new column "k clusters" to csv
    newfile = filename.split(".")[0]
    newfile+="_{}_clusters.csv".format(k)
    newfile = os.path.join(os.getcwd(), "Image_features", "HOG", newfile)
    oldfile = os.path.join(os.getcwd(), "Image_features", "HOG", filename)
    input = pd.read_csv(oldfile)
    col_name = "{} clusters".format(k)
    input[col_name] = preds
    input.to_csv(newfile, index=False)
    os.remove(oldfile)
    os.rename(newfile, oldfile)
    return sil


def hog_kmeans_linear(start, stop, filename):
    sil_scores = []
    for i in range(start, stop+1, 1):
        sil_scores.append(kmeans_hog(filename, i))
    root_folder = filename.split(".")[0]
    figname = root_folder + ".png"
    y = [x for x in range(start, stop+1)]
    plt.plot(sil_scores, y)
    plt.title("Silhouette Score for different values of k,\n{} dataset".format(filename))
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette Score")
    plt.savefig(os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, figname))
    print(sil_scores)


def classes_in_clusters(csv_file, savefile = None):
    if savefile is None:
        root_folder = csv_file.split(".")[0]
        fname = root_folder + "_class_stats.csv"
        savefile = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, fname)

    fd = open(savefile, "w+")
    df = load_hog_df(csv_file)
    classes = df['img_class']
    class_counts = classes.value_counts()   # can get the number of images in a class using class_counts.loc[<class>]
    # iterate over different k values
    for i in range(2, 44):
        num_clusters = i
        col_name = "{} clusters".format(i)
        clusters = df[col_name]
        cluster_counts = clusters.value_counts()    #number of images in a cluster = cluster_counts.loc[<cluster>]

        # dict representing % of total class images in a cluster.  Labels are class names as ints and values are dicts
        # where the label is the cluster name and the value is the percent of all class images in a that cluster. For
        # a given class A and cluster B, the value union_over_classes[A][B] represents
        # (# of imgs in class A and cluster B)/(# of images of class A)
        union_over_class = {}
        for j in range(43):
            union_over_class[j] = {}

        # dict representing % of total cluster images in a class.  Labels are cluster names as ints and values are
        # dicts where the label is the class name and the value is the percent of all cluster images of a certain class.
        # For a given class A and cluster B, the value unison_over_cluster[B][A] represents
        # (# of imgs in class A and cluster B)/(# of imgs in cluster B)
        union_over_cluster = {}
        for j in range(num_clusters):
            union_over_cluster[j] = {}

        # iterate over number of classes (43)
        for class_name in range(43):
            #rows of class j
            class_df = df[df['img_class'] == class_name]

            # iterate over clusters
            for cluster_name in range(num_clusters):
                # find # of images in class A and cluster B
                union_df = class_df[class_df[col_name] == cluster_name]
                union_count = len(union_df.index)
                percent_of_total_class_in_cluster = union_count/class_counts.loc[class_name]
                percent_of_cluster_comprising_class = union_count/cluster_counts.loc[cluster_name]
                union_over_class[class_name][cluster_name] = percent_of_total_class_in_cluster
                union_over_cluster[cluster_name][class_name] = percent_of_cluster_comprising_class

        # write to file
        fd.write(col_name + "\n")
        fd.write("% of total class images in cluster\n")
        fd.write("class/cluster")
        for j in range(num_clusters):
            fd.write(",cluster " + str(j))
        fd.write("\n")
        for key in union_over_class:
            fd.write("class " + str(key))
            for cluster in union_over_class[key]:
                fd.write("," + str(union_over_class[key][cluster]))
            fd.write("\n")

        fd.write("\n")

        fd.write("% of cluster made up of <class>\n")
        fd.write("cluster/class")
        for j in range(43):
            fd.write(",class " + str(j))
        fd.write("\n")
        for key in union_over_cluster:
            fd.write("cluster " + str(key))
            for clas in union_over_cluster[key]:
                fd.write("," + str(union_over_cluster[key][clas]))
            fd.write("\n")

        fd.write("\n\n")


    fd.close()


def shapes_in_clusters(csv_file, savefile = None):
    if savefile is None:
        root_folder = csv_file.split(".")[0]
        fname = root_folder + "_shape_stats.csv"
        savefile = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, fname)

    fd = open(savefile, "w+")
    df = load_hog_df(csv_file)

    shapes = ["circle", "triangle", "diamond", "inverted_triangle", "octagon"]

    # array where each index represents a class, and the value at that index represents the shape for that class
    class_shape = split_all("shape")

    classes = df['img_class']
    class_counts = classes.value_counts()  # can get the number of images in a class using class_counts.loc[<class>]

    # get number of images of each shape
    shape_counts = {"circle": 0, "triangle": 0, "diamond": 0, "inverted_triangle": 0, "octagon": 0}

    # dict where keys are shapes and values are arrays representing classes (as integers) of that shape
    shape_classes = {"circle": [], "triangle": [], "diamond": [], "inverted_triangle": [], "octagon": []}

    for class_name in range(43):
        shape = class_shape[class_name]
        shape_counts[shape] += class_counts.loc[class_name]
        shape_classes[shape].append(class_name)

    #all data of union_over_shape (discussed below)
    all_shape_data = {}

    # iterate over different k values
    for i in range(2, 44):
        num_clusters = i
        col_name = "{} clusters".format(i)
        clusters = df[col_name]
        cluster_counts = clusters.value_counts()    #number of images in a cluster = cluster_counts.loc[<cluster>]

        # dict representing % of total shape images in a cluster.  Labels are shapes as strings and values are dicts
        # where the label is the cluster name and the value is the percent of all shape images in a that cluster. For
        # a given shape A and cluster B, the value union_over_classes[A][B] represents
        # (# of imgs of shape A and cluster B)/(# of images of shape A)
        union_over_shape = {"circle": {}, "triangle": {}, "diamond": {}, "inverted_triangle": {}, "octagon": {}}

        # dict representing % of total cluster images that are a certain shape.  Labels are cluster names as ints and
        # values are dicts where the label is the shape and the value is the percent of all cluster images of that shape
        # For a given shape A and cluster B, the value unison_over_cluster[B][A] represents
        # (# of imgs of shape A and cluster B)/(# of imgs in cluster B)
        union_over_cluster = {}
        for j in range(num_clusters):
            union_over_cluster[j] = {}

        # iterate over shapes
        for shape_name in shapes:
            #rows with this shape
            shape_df = df[df['img_class'].isin(shape_classes[shape_name])]

            # iterate over clusters
            for cluster_name in range(num_clusters):
                # find # of images in shape A and cluster B
                union_df = shape_df[shape_df[col_name] == cluster_name]
                union_count = len(union_df.index)
                percent_of_total_shape_in_cluster = union_count/shape_counts[shape_name]
                percent_of_cluster_comprised_of_shape = union_count/cluster_counts.loc[cluster_name]
                union_over_shape[shape_name][cluster_name] = percent_of_total_shape_in_cluster
                union_over_cluster[cluster_name][shape_name] = percent_of_cluster_comprised_of_shape

        all_shape_data[num_clusters] = union_over_shape

        # write to file
        fd.write(col_name + "\n")
        fd.write("% of total shape images in cluster\n")
        fd.write("shape/cluster")
        for j in range(num_clusters):
            fd.write(",cluster " + str(j))
        fd.write("\n")
        for shape in shapes:
            fd.write(shape)
            for cluster in union_over_shape[shape]:
                fd.write("," + str(union_over_shape[shape][cluster]))
            fd.write("\n")

        fd.write("\n")

        fd.write("% of cluster made up of shape\n")
        fd.write("cluster/shape")
        for shape in shapes:
            fd.write("," + shape)
        fd.write("\n")
        for key in union_over_cluster:
            fd.write("cluster " + str(key))
            for shape in union_over_cluster[key]:
                fd.write("," + str(union_over_cluster[key][shape]))
            fd.write("\n")

        fd.write("\n\n")

    # calculate metrics by number of clusters
    cluster_metrics = {}

    # iterate over number of clusters (43)
    for num_clusters in range(2, 44):
        total_metric = 0
        for shape in shapes:
            shape_vector = list(all_shape_data[num_clusters][shape].values())
            max_ind = shape_vector.index(max(shape_vector))
            ideal = [0 if x!=max_ind else 1 for x in range(len(shape_vector))]
            diff = sum([abs(shape_vector[x] - ideal[x])**2 for x in range(len(shape_vector))])
            # average diff by number of clusters
            total_metric +=diff/num_clusters
        cluster_metrics[num_clusters] = total_metric

    fd.write("Metrics\n")
    for key in cluster_metrics:
        fd.write("{} clusters, {}\n".format(key, cluster_metrics[key]))

    fd.close()





def strip_trailing_commas(filename, newfilename):
    filename = os.path.join(os.getcwd(), "Image_features", filename)
    newfilename = os.path.join(os.getcwd(), "Image_features", newfilename)
    file = open(filename, 'r')
    target = open(newfilename, 'w')
    for line in file:
        target.write(line[:-1].rstrip(',') + "\n")
    file.close()
    target.close()


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


def gather_hog(test_folder, bins = 12):
    # given a folder with subfolders named as the sign classes, this function will traverse through the subfolders
    # and gather HOG of each image, and return an array of 3-tuples of the format:
    # (<file name>, <class name>, <[array of HOG]>)
    img_data = []
    classes = os.listdir(test_folder)
    for sign in classes:
        sign_folder = os.path.join(test_folder, sign)
        for file in os.listdir(sign_folder):
            img_file = os.path.join(test_folder, sign, file)
            img_data.append((file, sign, get_hog(img_file, orientations=bins)))
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


def get_max_color_length(img_file, threshold = 25, size = (124, 124), color_limits = (0, 0), display = False):
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

    if display:
        display_max_color_len(image, max_color_length, x_start, y_start, max_direction)

    return max_color_length, x_start, y_start, max_direction, max_color


def display_max_color_len(image, max_color_length, x_start, y_start, max_direction):
    new_im = image.copy()
    while(max_color_length>0):
        new_im.putpixel((x_start, y_start), (204,255,0))
        max_color_length -= 1
        x_start, y_start = get_next_pixel(x_start, y_start, 10000, 10000, max_direction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(new_im)
    ax2.set_title('Max Color Line')
    plt.show()


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


def avg_rgb(img_file, plot = False, hist = False):
    # adapted from https://datacarpentry.org/image-processing/aio/index.html
    # returns a 3-tuple: (red average, green average, blue average)
    # if hist = True, then returns a histogram of RBG formatted as a 3-tuple(red array, green array, blue array)
    image = skimage.io.imread(fname=img_file)

    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    avgs = {}
    r = list(np.concatenate(image[:, :, 0]))
    b = list(np.concatenate(image[:, :, 1]))
    g = list(np.concatenate(image[:, :, 2]))

    if not hist:
        avgs["r"] = sum(r) / len(r)
        avgs["g"] = sum(g) / len(g)
        avgs["b"] = sum(b) / len(b)

    if plot:
        # create the histogram plot, with three lines, one for
        # each color
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(image)
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
    if hist:
        return (r, g, b)
    return avgs["r"], avgs["g"], avgs["b"]


def get_hog(im_file, orientations=12, pix_per_cell = (224, 224), cells_per_block=(1,1), visualize=False,
            saturating = 0.7, norm = 1):
    # pix_per_cell set to image size so that histogram is computed over entire image
    # saturating - limit each histogram value to a max of <saturating> (can be turned off if set to None)
    # norm - normalize so sum of feature vector is <norm> (can be turned off if set to None)
    image = Image.open(im_file)
    image = image.resize((224, 224))
    if visualize:
        _, hog_image = hog(image, orientations=orientations, pixels_per_cell=pix_per_cell,
                           cells_per_block=cells_per_block, visualize=visualize, multichannel=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
        return None
    else:
        feature_vect = list(hog(image, orientations=orientations, pixels_per_cell=pix_per_cell, #block_norm='L1',
                           cells_per_block=cells_per_block, visualize=visualize, multichannel=True,
                           feature_vector=True))
        if saturating is not None:
            for i in range(len(feature_vect)):
                if feature_vect[i] > saturating:
                    feature_vect[i] = saturating
        if norm is not None:
            total = sum(feature_vect)
            for i in range(len(feature_vect)):
                feature_vect[i] = feature_vect[i]/total
        return feature_vect


def test_hog(astronaut = False, visualize = True):
    if astronaut:
        im = data.astronaut()
    else:
        im = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
        im = os.path.join(os.getcwd(), "vertical_lines_test.jpg")
    a = get_hog(im, visualize=visualize)
    if not visualize:
        print("Feature Vector:\n{}\nLength of feature vector:\n{}".format(a, len(a)))
        print(sum(a))
        return a


def test_rgb(plot=True):
    im = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
    a = avg_rgb(im, plot=plot)
    print("Avg RGB:\n{}".format(a))


def test_rgb_2():
    im = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
    image = Image.open(im)
    image = image.resize((224, 224))
    a = image.histogram()
    r = a[:256]
    g = a[256:512]
    b = a[512:]
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(image)
    plt.xlim([0, 256])
    plt.plot(r, 'r')
    plt.plot(g, 'g')
    plt.plot(b, 'b')
    plt.legend()
    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()
    print(a)


def test_max_color_len(color_lims = (0,0)):
    im = os.path.join(os.getcwd(), "Train", "22", "00022_00000_00000.png")
    get_max_color_length(im, display=True, color_limits=color_lims)


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


def test_dominat_colors(num = 5):
    im = os.path.join(os.getcwd(), "Train", "22", "00022_00000_00000.png")
    get_dominant_colors(im, show_chart=True, number_of_colors=num)


if __name__ == '__main__':
    #file = os.path.join(os.getcwd(), 'Debug', '00', '03922.png')
    # file = os.path.join(os.getcwd(), 'Debug', '00', '06946.png')
    # print(get_max_color_length(file))
    #print(get_dominant_colors(file, 5, True))
    #print(gather_image_features(file))
    #store_csv_hog(img_folder=os.path.join(os.getcwd(), "small_test_dataset", "Train"), validation=False)

    #kmeans_hog("12hog_img_features_train_val_2021-02-22.csv", 2)
    #kmeans_hog("12hog_img_features_Train_2021-02-23.csv", 3)
    #hog_kmeans_linear(2, 10, "12hog_img_features_Train_2021-02-23.csv")
    #hog_kmeans_linear(43, 43, "12hog_img_features_train_val_2021-02-22.csv")
    #classes_in_clusters("12hog_img_features_train_val_2021-02-22.csv")
    shapes_in_clusters("12hog_img_features_train_val_2021-02-22.csv")
    #print(gather_test_data(test_folder))
    #test_store_csv()
    # csv = os.path.join(os.getcwd(), "Image_features", "img_features_GTSRB_ResNet_2020-12-29.csv")
    # a = normalize_csv(csv)
    # print(a)
    #test_hog(visualize=False)
    #test_rgb()
    #test_max_color_len(color_lims=(0,0))
    #test_dominat_colors(10)
    #strip_trailing_commas("12hog_img_features_GTSRB_ResNet_2021-02-22.csv", "12hog_img_features_train_val_2021-02-22.csv")