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
import pickle
from tree_helper import split_all
from kneebow.rotor import Rotor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score

# length of feature vector
FEATURE_LENGTH = 22

def save_kmeans(kmeans, root_folder, num_clusters, centers, filename = None):
    # saves a kmeans model and cluster centers
    # cluster centers is a numpy 2d array
    kmeans_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, "kmeans_models")
    if not os.path.exists(kmeans_folder):
        os.mkdir(kmeans_folder)
    if filename is None:
        filename = "kmeans_{}_clusters.pkl".format(str(format_two_digits(num_clusters)))
    cluster_filename = "kmeans_{}_clusters_centers.txt".format(format_two_digits(num_clusters))
    fname = os.path.join(root_folder, kmeans_folder, filename)
    pickle.dump(kmeans, open(fname, "wb"))
    fname = os.path.join(root_folder, kmeans_folder, cluster_filename)
    np.savetxt(fname, centers, fmt='%f')
    #print(centers)
    print("Saved kmeans model with {} clusters at {}".format(num_clusters, fname))


def load_kmeans(root_folder, num_clusters = 2, filename = None):
    # loads a kmeans model and clusters centers and returns a tuple (kmeans model, cluster centers) where
    # cluster centers is a numpy 2d array
    num_clusters = format_two_digits(num_clusters)
    if filename is None:
        filename = "kmeans_{}_clusters.pkl".format(str(num_clusters))
    centers_filename = "kmeans_{}_clusters_centers.txt".format(str(num_clusters))
    fname = os.path.join(root_folder, "kmeans_models", filename)
    kmeans_model = pickle.load(open(fname, "rb"))
    fname = os.path.join(root_folder, "kmeans_models", centers_filename)
    centers = np.loadtxt(fname, dtype=float)
    return kmeans_model, centers


def load_all_kmeans(root_folder_name):
    # takes a root folder name and traverses the kmeans_models subdirectory and returns an array of tuples
    # (kmeans models, cluster centers) one tuple for each model in the subdirectory

    all_models = []
    root_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name)
    kmeans_folder = os.path.join(root_folder, "kmeans_models")
    files = os.listdir(kmeans_folder)
    for f in files:
        #only call load_kmeans on .pkl files
        if f.find("txt")>=0:
            continue
        # get number of clusters
        num_clusters = ""
        for char in f:
            if char.isdigit():
                num_clusters +=char
        num_clusters = format_two_digits(int(num_clusters))
        model_path = os.path.join(kmeans_folder, f)
        all_models.append(load_kmeans(root_folder, num_clusters=num_clusters, filename=model_path))
    return all_models


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


def store_csv_hog(bins = 20, img_folder = None, filename = None, save_folder = None, validation = True,
                  pix_per_cell = (112,112)):
    # gathers the HOG for each image in <img_folder> and stores this data in csv format under a
    # file named <filename> and stored in <save_folder>
    #
    # if <validation> is True, then <img_folder> should be a root directory containing subfolders of "Train"
    # and "Validation" and all of the images in these subfolders will be stored in the csv.  This feature
    # was added because the neural network code partitions the csv into training and validation data
    total_bins = bins
    if pix_per_cell[0] != 224:
        factor = 224/pix_per_cell[0]
        total_bins *= int(factor**2)

    if img_folder == None:
        img_folder = os.getcwd()
    if filename == None:
        date = str_date()
        test_folder_abbreviated = os.path.split(img_folder)[1]
        filename = str(total_bins) + "hog_img_features_" + str(test_folder_abbreviated) + "_" + date + ".csv"
    if save_folder == None:
        root_folder = filename.split(".")[0]
        save_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    if validation:
        train_folder = os.path.join(img_folder, "Train")
        val_folder = os.path.join(img_folder, "Validation")
        img_data = gather_hog(train_folder, bins, pix_per_cell = pix_per_cell)
        img_data.extend(gather_hog(val_folder, bins, pix_per_cell = pix_per_cell))
    else:
        img_data = gather_hog(img_folder, bins, pix_per_cell = pix_per_cell)

    csv_filename = os.path.join(save_folder, filename)
    f = open(csv_filename, "w")

    f.write("filename,img_class")

    for i in range(total_bins):
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
    return filename


def collect_test_data_hog():
    img_folder = os.path.join(os.getcwd(), "Test")
    store_csv_hog(img_folder=img_folder, validation=False)


def use_models_to_predict_new(old_filename, new_filename):
    # takes the kmeans models in the directory with old_filename and uses the kmeans models to predict clusters
    # for new_filename
    root_folder = old_filename.split(".")[0]
    root_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder)
    # array of tuples (kmeans_model, cluster_centers for that model) where cluster_centers is a 2d numpy array
    k_models_centers = load_all_kmeans(root_folder)
    end = len(k_models_centers)+1
    hog_kmeans_linear(2, end, new_filename, models = k_models_centers, save=False)


def distances_to_cluster_centers(feature_vector, cluster_centers, label = None):
    # takes a feature vector array and cluster centers array (2d of shape (num_clusters, length of feature vector))
    # and returns an array of len(num_clusters) where each index is the distance from each of the cluster centers

    # if label is not None, it should be an integer representing the cluster for the sample represented by the
    # feature vector that was chosen in k-means clustering.  This function will double check that the label
    # matches the shortest distance to one of the clusters

    feature_vector = np.array(feature_vector)
    cluster_centers = np.array(cluster_centers)
    distances_to_centers = []

    for x in cluster_centers:
        dist = np.linalg.norm(feature_vector-x)
        #dist = sum(abs(a-b) for a,b in zip(list(feature_vector),list(x)))
        distances_to_centers.append(dist)

    if label is not None:
        distances = list(distances_to_centers)
        closest_cluster = distances.index(min(distances))
        if closest_cluster != label:
            print("Error in mapping cluster centers to label, label was {} and closest cluster was {}".format(
                label, closest_cluster
            ))

    return distances_to_centers


def kmeans_model_predict_one(feature_vector, kmeans_model):
    # uses a kmeans model to predict the cluster for a single feature vector.  Returns a cluster name as an integer
    # feature vector should be a list
    feature_vector = np.array(feature_vector)
    feature_vector = np.reshape(feature_vector, (1, -1))
    pred_cluster = kmeans_model.predict(feature_vector)
    return pred_cluster


def cluster_pred_vector(pred_vector):
    # takes a prediction vector and uses k-means clustering with k=2 on the prediction vector.
    # this results in 2 sets: classes that are in the prediction and classes that are excluded from the prediction

    pred_vector_np = np.array(pred_vector).reshape((-1, 1))

    kmeans = KMeans(n_clusters=2, random_state=0)
    preds = kmeans.fit_predict(pred_vector_np)
    centers = kmeans.cluster_centers_.reshape(1, -1).tolist()[0]
    included_cluster = centers.index(max(centers))
    included_classes = []
    for i in range(len(preds)):
        if preds[i] == included_cluster:
            included_classes.append(i)
    return included_classes


def test_all_cluster_pred_vector(root_folder_name = None, k_value = 3, attribute = "hog", pred_folder_name = None):

    if root_folder_name is None:
        root_folder_name = "80hog_img_features_GTSRB_ResNet_2021-03-11"
    if pred_folder_name is None:
        pred_folder_name = root_folder_name
    if attribute == "hog":
        start_col = 2
        end_col = 82
    else:
        print("Invalid attribute name in unsupervised_predict_all, {}".format(attribute))
        return -1

    root_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name)
    pred_folder = os.path.join(os.getcwd(), "Image_features", "HOG", pred_folder_name)
    csv_name = pred_folder_name + ".csv"

    original_csv_df = load_hog_df(csv_name)
    num_rows = len(original_csv_df.index)
    total_correct = 0

    # get purity of each cluster (dict), and class balance of that cluster (dict)
    purities, class_balance = calculate_cluster_impurity(root_folder_name, k_value)

    # get cluster_centers for <k_value>
    model, centers = load_kmeans(root_folder, num_clusters=k_value)

    # 2d array where 1st dimension is a sample and 2nd dimension is an array of
    # img filename, img class, [prediction vector over all classes]
    all_preds = []

    lengths_of_included_classes = []

    for i in range(num_rows):
        if(i%1000 == 0):
            print("{} rows to go".format(num_rows-i))
        row_data = original_csv_df.iloc[i]
        img_name_and_class = list(row_data)[:2]
        feature_vector = list(row_data)[start_col:end_col]

        # get cluster prediction from the model
        cluster_pred = kmeans_model_predict_one(feature_vector, model)

        # get distance from feature vector to each centroid
        distance_to_centers = distances_to_cluster_centers(feature_vector, centers, label=cluster_pred)

        # use a prediction function to output a prediction vector over all classes
        pred_vector = prediction_vector_unsupervised(distance_to_centers, purities, class_balance, arr = True,
                                                     increase_diffs=2.5)

        img_class = img_name_and_class[1]
        included_classes = cluster_pred_vector(pred_vector)
        lengths_of_included_classes.append(len(included_classes))
        if img_class in included_classes:
            total_correct += 1

    avg_len  = sum(lengths_of_included_classes)/len(lengths_of_included_classes)
    print(lengths_of_included_classes)
    print("Avg length of included classes {}, complement is {}".format(avg_len, 43-avg_len))
    print("Total: {}/{} images correct, {:.4f}%".format(total_correct, num_rows, 100 * total_correct / num_rows))


def unsupervised_predict_all(root_folder_name, k_value, save_filename = None, attribute = "hog", num_classes = 43,
                             top_n=None, predict_folder_name = None):
    # iterates through the main csv file from root_folder and creates a new csv_file called
    # unsupervised_class_predictions.csv which represents the prediction vector of each feature vector
    # top_n: array of top_n accuracies to compute.  They will get stored in a txt file top_n_accuracies.txt
    # root_folder name: where the kmeans model and class_balance are derived from
    # predict_folder: will use the stuff in root_folder to predict for a set of feature vectors in predict_folder

    if top_n is None:
        top_n = [1, 5, 10, 20, 25]
    if predict_folder_name is None:
        predict_folder_name = root_folder_name
    if attribute == "hog":
        start_col = 2
        end_col = 82
    else:
        print("Invalid attribute name in unsupervised_predict_all, {}".format(attribute))
        return -1

    root_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name)
    predict_folder = os.path.join(os.getcwd(), "Image_features", "HOG", predict_folder_name)
    csv_name = predict_folder_name + ".csv"
    #csv = os.path.join(predict_folder_name, csv_name)

    if save_filename is None:
        save_filename = "unsupervised_class_predictions.csv"

    original_csv_df = load_hog_df(csv_name)
    num_rows = len(original_csv_df.index)

    # get purity of each cluster (dict), and class balance of that cluster (dict)
    purities, class_balance = calculate_cluster_impurity(root_folder_name, k_value)

    # get cluster_centers for <k_value>
    model, centers = load_kmeans(root_folder, num_clusters=k_value)

    # 2d array where 1st dimension is a sample and 2nd dimension is an array of
    # img filename, img class, [prediction vector over all classes]
    all_preds = []

    for i in range(num_rows):
        if(i%1000 == 0):
            print("{} rows to go".format(num_rows-i))
        row_data = original_csv_df.iloc[i]
        img_name_and_class = list(row_data)[:2]
        feature_vector = list(row_data)[start_col:end_col]

        # get cluster prediction from the model
        cluster_pred = kmeans_model_predict_one(feature_vector, model)

        # get distance from feature vector to each centroid
        distance_to_centers = distances_to_cluster_centers(feature_vector, centers, label=cluster_pred)

        # use a prediction function to output a prediction vector over all classes
        pred_vector = prediction_vector_unsupervised(distance_to_centers, purities, class_balance, arr = True,
                                                     increase_diffs=2.5)

        img_name_and_class.extend(pred_vector)
        all_preds.append(img_name_and_class)

    top_n_accuracies(all_preds, predict_folder, top_n, num_classes=num_classes)

    # write results to csv
    save_path = os.path.join(predict_folder, save_filename)
    f = open(save_path, 'w+')

    # header row
    f.write("img,class")
    for i in range(num_classes):
        f.write(",class_{}".format(i))
    f.write("\n")

    # data
    for i in range(len(all_preds)):
        for j in range(len(all_preds[i])):
            if j < len(all_preds[i]) -1:
                f.write(str(all_preds[i][j]) + ",")
            elif i < len(all_preds) -1:
                f.write(str(all_preds[i][j]) + "\n")
            else:
                f.write(str(all_preds[i][j]))

    f.close()

def unsupervised_predict_one(feature_vector, root_folder_name = "80hog_img_features_GTSRB_ResNet_2021-03-11",
                             k_value = 3):
    # uses the csv on the Train dataset to get a prediction vector over all classes
    root_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name)

    # get purity of each cluster (dict), and class balance of that cluster (dict)
    purities, class_balance = calculate_cluster_impurity(root_folder_name, k_value)

    # get cluster_centers for <k_value>
    model, centers = load_kmeans(root_folder, num_clusters=k_value)

    # get cluster prediction from the model
    cluster_pred = kmeans_model_predict_one(feature_vector, model)

    # get distance from feature vector to each centroid
    distance_to_centers = distances_to_cluster_centers(feature_vector, centers, label=cluster_pred)

    # use a prediction function to output a prediction vector over all classes
    return prediction_vector_unsupervised(distance_to_centers, purities, class_balance, arr=True, increase_diffs=2.5)

def top_n_accuracies(all_preds, root_folder, top_n = [1, 5, 10, 20, 25], num_classes = 43):
    # computes the top_n accuracy for the unsupervised predict_all function for each value of n in top_n
    # results are stored in top_n_accuracies.txt under root_folder
    # all_preds: 2d array from unsupervised_predict_all where 1st dimension is a sample and 2nd dimension is an array
    # of img filename, img class, [prediction vector over all classes]

    total_samples = len(all_preds)
    top_n = sorted(top_n)

    # dict representing the total correct for each value of n
    total_correct = {}
    for i in top_n:
        total_correct[i] = 0

    for row in all_preds:
        true_class = row[1]
        pred_vector = row[2:]
        if len(pred_vector) != num_classes:
            print("Error in top_n_accuracies:\nThere are {} classes but the length of the prediction vector was {}"
                  .format(num_classes, len(pred_vector)))
            return

        # convert prediction vector to an array of tuples (class, confidence in that class)
        pred_tuple = [(i, pred_vector[i]) for i in range(len(pred_vector))]

        # sort pred_tuple by class confidence
        sorted_preds = sorted(pred_tuple, key=lambda x: x[1], reverse=True)

        # only need to see if true class is within the max top_n
        true_class_index = None
        for i in range(top_n[len(top_n)-1]):
            if sorted_preds[i][0] == true_class:
                true_class_index = i
                break

        if true_class_index is None:
            continue
        else:
            for n in top_n:
                if true_class_index <= n:
                    total_correct[n] += 1

    # save to a file
    save_path = os.path.join(root_folder, "top_n_accuracies.txt")
    f = open(save_path, 'w+')

    for n in total_correct:
        f.write("Top-{} accuracy:\n{}/{} images correct, {:.4f}%\n\n".format(n, total_correct[n], total_samples,
                                                                             100*total_correct[n]/total_samples))
    f.close()

def hog_predict_classes_one(feature_vector, k_value, root_folder, hard_prediction = False):
    # given the hog feature vector for one image, give a prediction over all classes for that feature vector
    # hard_prediction - for each cluster, take the class balance inside that cluster and cluster it with k=2.  Take
    #                   only the classes in the cluster with a higher centroid as "included" in this cluster.  If this
    #                   value is false, then take all classes for each cluster


    # get cluster_centers for <k_value>
    model, centers = load_kmeans(root_folder, num_clusters=k_value)

    # get cluster prediction from the model
    cluster_pred = kmeans_model_predict_one(feature_vector, model)

    # get distance from feature vector to each centroid
    distance_to_centers = distances_to_cluster_centers(feature_vector, centers, label=cluster_pred)

    # get purity of each cluster (dict), and class balance of that cluster (dict)
    purities, class_balance = calculate_cluster_impurity(root_folder, k_value)

    # use a prediction function to output a prediction vector over all classes
    pred_vector = prediction_vector_unsupervised(distance_to_centers, purities, class_balance)

    return pred_vector


def prediction_vector_unsupervised(distance_to_centers, cluster_purities, class_balance, arr = True,
                                   increase_diffs = 1.0):
    # for one feature vector, output a prediction vector over all classes
    # if array is true, returns an array where the indices represent the classes and values represents confidence,
    # else returns a dict where the keys represent the classes and the values represent confidence
    # increase diffs: integer - will raise all elements of the distance to centers vector by a power of <increase_diffs>
    # before normalization in order to increase the difference between the cluster centers.

    classes = list(class_balance[0].keys())

    # one coefficient per cluster, factoring in distance to each cluster and cluster purity
    # it is better to have high purity but low distance to a cluster, therefore I normalize the purities wrt each other
    # and normalize the distances to each cluster wrt each other, then take the complement of the distance to the clusters
    # so that higher is better
    # purities are already normalized
    distance_to_centers = [(1/x)**increase_diffs for x in distance_to_centers]
    total_dist = sum(distance_to_centers)
    #distance_to_centers = list(np.array(distance_to_centers)/np.linalg.norm(np.array(distance_to_centers)))
    distance_to_centers = [(x/total_dist) for x in distance_to_centers]

    cluster_coefficients = {}
    for i, cluster_name in enumerate(cluster_purities):
        cluster_coefficients[cluster_name] = cluster_purities[cluster_name] * distance_to_centers[i]

    prediction_vector = {}

    for cluster_name in cluster_purities:
        for class_name in classes:
            if class_name in prediction_vector:
                prediction_vector[class_name] += cluster_coefficients[cluster_name] * class_balance[cluster_name][class_name]
            else:
                prediction_vector[class_name] = cluster_coefficients[cluster_name] * class_balance[cluster_name][
                    class_name]

    # normalize pred_vector
    total = 0
    for key in prediction_vector:
        total += prediction_vector[key]
    for key in prediction_vector:
        prediction_vector[key] = prediction_vector[key]/total

    if arr:
        # prediction vector is not sorted by class
        pred_array = []
        for key in prediction_vector:
            pred_array.append((key, prediction_vector[key]))
        # sort by class
        pred_array = sorted(pred_array, key=lambda x: x[0])
        # get rid of tuple (class_name, confidence) to just be an array of confidences
        pred_array = [x[1] for x in pred_array]
        return pred_array

    return prediction_vector


def predict_shape(original_csv, save_filename = None):
    # takes a csv file from <root_folder> (one that has the cluster predictions) and takes
    # information collected from a shapes_in_clusters csv to predict the shape of an image.  Creates a new
    # csv called shape_predictions where the columns are
    # img path, img class, img shape, shape prediction (string)
    # for now this shapes_in_clusters info is hard-coded, goes from 2 to 18 clusters
    # returns the prediction accuracy
    # *** original csv is assumed to be 80hog_img_features_GTSRB_ResNet_2021-03-11.csv because the function
    # cluster_preds_to_shape_preds below is hard coded for this

    def cluster_preds_to_shape_preds(cluster_preds):
        # cluster_preds is a dict of the cluster predictions for one image, going from 2 to 18 clusters
        # the keys are the number of clusters as ints and the values are the cluster prediction for that number
        # of clusters
        # this function maps the cluster_arr to one of [cdo, triangle, inverse_triangle] where
        # cdo represents circle, diamond, or octagon (currently cannot distinguish between these)

        if cluster_preds[18] == 16 or cluster_preds[19] == 15 or cluster_preds[16] == 10:
            return "inverted_triangle"
        if cluster_preds[18] == 0 or cluster_preds[19] == 13 or cluster_preds[15] == 5:
            return "cdo"
        if cluster_preds[19] == 11:
            return "triangle"
        if (cluster_preds[3] == 0 or cluster_preds[4] == 0 or cluster_preds[5] == 3 or
            cluster_preds[6] == 5 or cluster_preds[7] == 6 or cluster_preds[7] == 4 or
            cluster_preds[8] == 4 or cluster_preds[8] == 6 or cluster_preds[9] == 1 or
            cluster_preds[9] == 3 or cluster_preds[18] == 12 or cluster_preds[18] == 10 or cluster_preds[14] == 4):
            return "triangle"
        if(cluster_preds[5] == 0 and (cluster_preds[6] == 1 or cluster_preds[7] == 6
                                      or cluster_preds[8] == 0 or cluster_preds[9] == 8 or
                                    cluster_preds[10] == 9)):
            return "inverted_triangle"
        else:
            return "cdo"

    if save_filename is None:
        save_filename = "shape_predictions.csv"

    use_columns = ["filename", "img_class"]
    for i in range(2, 20):
        use_columns.append("{} clusters".format(i))

    original_csv_df = load_hog_df(original_csv, usecols=use_columns)

    root_folder = original_csv.split(".")[0]
    save_path = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, save_filename)
    f = open(save_path, 'w+')
    f.write("img,class,predicted_shape,true_shape,correct")

    # array where each index represents a class, and the value at that index represents the attribute label for that
    # class
    class_attributes = split_all("shape")

    total_correct = 0
    shapes_correct = {"cdo": 0, "triangle": 0, "inverted_triangle": 0}
    shapes_count = {"cdo": 0, "triangle": 0, "inverted_triangle": 0}

    num_rows = original_csv_df.count()[0]
    cluster_cols = [i for i in range(2, len(use_columns))]
    for i in range(num_rows):
        correct = 0

        # get a row of cluster predictions
        data = original_csv_df.iloc[[i], cluster_cols]
        cluster_preds_arr = list(data.iloc[0])
        cluster_preds = {}
        for count,x in enumerate(cluster_cols):
            cluster_preds[x] = cluster_preds_arr[count]
        pred_shape = cluster_preds_to_shape_preds(cluster_preds)

        # get img file name and img class
        data = original_csv_df.iloc[[i], [0,1]]
        data = list(data.iloc[0])
        img_name = data[0]
        true_class = data[1]
        true_shape = class_attributes[true_class]

        if true_shape == "triangle":
            shapes_count["triangle"] += 1
            if pred_shape == "triangle":
                total_correct += 1
                shapes_correct["triangle"] += 1
                correct = 1
        elif true_shape == "inverted_triangle":
            shapes_count["inverted_triangle"] += 1
            if pred_shape == "inverted_triangle":
                total_correct += 1
                shapes_correct["inverted_triangle"] +=1
                correct = 1
        else:
            shapes_count["cdo"] += 1
            if pred_shape == "cdo":
                total_correct += 1
                shapes_correct["cdo"] += 1
                correct = 1
        f.write("\n{},{},{},{},{}".format(img_name, true_class, pred_shape, true_shape, correct))

    f.close()
    print("total correct:\t\t{}/{}, {:4f}%".format(total_correct, num_rows, 100 * total_correct/num_rows))
    for key in shapes_count:
        print("{}:\t\t{}/{} correct, {:4f}%".format(key, shapes_correct[key],shapes_count[key],
                                             100 * shapes_correct[key]/shapes_count[key]))


def load_hog_df(filename, usecols = None):
    # loads a hog csv from store_csv_hog into a dataframe
    root_folder = filename.split(".")[0]
    filename = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, filename)
    if usecols is not None:
        df = pd.read_csv(filename, usecols = usecols)
        return df
    else:
        df = pd.read_csv(filename)
        return df


def kmeans_hog(filename, k, kmeans_model = None, save = True):
    # takes a csv of HOG data and uses k-means clustering with <k> clusters to assign cluster labels to each row/entry
    # in the csv.  These labels are added as another column in the csv.

    # kmeans_model - tuple of (kmeans_model, cluster centers for that model), where cluster centers is a 2d numpy array
    #   this is used when kmeans on one csv is used to predict for another csv

    print("Using kmeans clustering on {} with {} clusters".format(filename, k))
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
    if kmeans_model is None:
        kmeans = KMeans(n_clusters=k, random_state=0)
        preds = kmeans.fit_predict(a)
        labels = kmeans.labels_
    else:
        kmeans = kmeans_model[0]
        original_centers = kmeans_model[1]
        preds = kmeans.predict(a)
        labels = preds
    cost = kmeans.inertia_


    # if kmeans_model is not None:
    #     #---------- map the new centers to the original centers ----------
    #     new_centers = kmeans.cluster_centers_
    #     print(new_centers)
    #
    #     # dict where keys are new_centers (ints) and values are original_centers (ints)
    #     mapping = {}
    #
    #     for i, new in enumerate(new_centers):
    #         for j, orig in enumerate(original_centers):
    #             if new.all() == orig.all():
    #                 print("found")
    #                 mapping[i] = j
    #
    #     for i in range(len(preds)):
    #         preds[i] = mapping[preds[i]]

    sil = silhouette_score(a, labels, metric='euclidean')
    # write new column "k clusters" to csv
    newfileroot = filename.split(".")[0]
    newfile = newfileroot + "_{}_clusters.csv".format(format_two_digits(k))
    newfile = os.path.join(os.getcwd(), "Image_features", "HOG", newfileroot, newfile)
    oldfile = os.path.join(os.getcwd(), "Image_features", "HOG", newfileroot, filename)
    input = pd.read_csv(oldfile)
    col_name = "{} clusters".format(k)
    input[col_name] = preds
    input.to_csv(newfile, index=False)
    os.remove(oldfile)
    os.rename(newfile, oldfile)
    if save:
        centers = kmeans.cluster_centers_
        # save kmeans model
        save_kmeans(kmeans, newfileroot, k, centers)
    return sil, cost


def hog_kmeans_linear(start, stop, filename, models = None, save = True):
    # uses k means clustering with a linear sweep of <start> to <stop> clusters on the HOG data in <filename>
    # each cluster value k will be a new column appended to <filename> and the entries of that column will
    # represent the class prediction for the images

    # filename: the name of a csv file in os.cwd()/Image_features/HOG/<root_folder of the hog data>
    # models (array of tuples (kmeans model, cluster centers for that model) where cluster centers is a 2d numpy array):
    #   if not none, then this is an array of precomputed kmeans models, and these models will be passed to kmeans_hog()
    #   as the kmeans prediction models
    sil_scores = []
    costs = []
    for i in range(start, stop+1, 1):
        if models is not None:
            kmeans_model = models[i-start][0]
            sil, cost = kmeans_hog(filename, kmeans_model.n_clusters, kmeans_model=models[i-start], save = save)
            sil_scores.append(sil)
            costs.append(cost)
        else:
            sil, cost = kmeans_hog(filename, i, save = save)
            sil_scores.append(sil)
            costs.append(cost)
    root_folder = filename.split(".")[0]
    plot_sil_scores(root_folder, sil_scores=sil_scores, save=True)
    plot_cost_scores(root_folder, cost_scores=costs, save=True)


def classes_in_clusters(csv_file, savefile = None, save = True, exclusive_clusters = None):
    # analyzes the presence of different classes in the clusters and the distribution of classes over the clusters,
    # saves this analysis in a new csv file

    # returns:
    # dict representing % of total cluster images in a class.  Labels are cluster names as ints and values are
    # dicts where the label is the class name and the value is the percent of all cluster images of a certain class.
    # For a given class A and cluster B, the value unison_over_cluster[B][A] represents
    # (# of imgs in class A and cluster B)/(# of imgs in cluster B)

    # save (bool) whether or not to save in a csv (will overwrite existing csv if this function has been called already)
    # exclusive_clusters (array of ints representing clusters) the clusters to calculate

    if savefile is None:
        root_folder = csv_file.split(".")[0]
        fname = root_folder + "_class_stats.csv"
        savefile = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder, fname)

    if save:
        fd = open(savefile, "w+")
    df = load_hog_df(csv_file)
    classes = df['img_class']
    class_counts = classes.value_counts()   # can get the number of images in a class using class_counts.loc[<class>]

    return_value = None

    min_k = ""
    max_k = ""
    col_names = list(df.columns)
    for name in col_names:
        if name.find("clusters") >= 0:
            i = 0
            while (name[i].isdigit()):
                min_k += name[i]
                i += 1
            min_k = int(min_k)
            break
    name = col_names[len(col_names) - 1]
    i = 0
    while (name[i].isdigit()):
        max_k += name[i]
        i += 1
    max_k = int(max_k)


    # iterate over different k values
    for i in range(min_k, max_k):
        if exclusive_clusters is not None:
            if i not in exclusive_clusters:
                continue
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
                return_value = union_over_cluster

        if save:
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


    if save:
        fd.close()

    return return_value, union_over_class


def shapes_in_clusters(csv_file, savefile = None):
    # similar to classes_in_clusters, but analyzes the presence of shapes instead
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

    min_k = ""
    max_k = ""
    col_names = list(df.columns)
    for name in col_names:
        if name.find("clusters")>=0:
            i = 0
            while(name[i].isdigit()):
                min_k += name[i]
                i+=1
            min_k = int(min_k)
            break
    name = col_names[len(col_names)-1]
    i =0
    while(name[i].isdigit()):
        max_k += name[i]
        i+=1
    max_k = int(max_k)

    # iterate over different k values
    for i in range(min_k, max_k):
        print("k value: {}".format(i))
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
            print(shape_name)
            #rows with this shape
            shape_df = df[df['img_class'].isin(shape_classes[shape_name])]

            # iterate over clusters
            for cluster_name in range(num_clusters):
                print(cluster_name)
                # if num_clusters == 11 and shape_name == "circle" and cluster_name == 2:
                #     print("hello")
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

    # iterate over number of clusters
    for num_clusters in range(min_k, max_k):
        total_metric = 0
        for shape in shapes:
            shape_vector = list(all_shape_data[num_clusters][shape].values())
            max_ind = shape_vector.index(max(shape_vector))
            ideal = [0 if x!=max_ind else 1 for x in range(len(shape_vector))]
            diff = sum([abs(shape_vector[x] - ideal[x])**2 for x in range(len(shape_vector))])
            # average diff by number of clusters
            total_metric +=diff#/num_clusters
        cluster_metrics[num_clusters] = total_metric

    fd.write("Metrics\n")
    for key in cluster_metrics:
        fd.write("{} clusters, {}\n".format(key, cluster_metrics[key]))

    fd.close()


def strip_trailing_commas(filename, newfilename):
    # strips the trailing commas from each line in a file
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


def gather_hog(test_folder, bins = 12, pix_per_cell = (112, 112), verbose = True):
    # given a folder with subfolders named as the sign classes, this function will traverse through the subfolders
    # and gather HOG of each image, and return an array of 3-tuples of the format:
    # (<file name>, <class name>, <[array of HOG]>)
    if verbose:
        print("Computing HOG for ...{}".format(test_folder[:70]))
    img_data = []
    classes = os.listdir(test_folder)
    for sign in classes:
        if verbose:
            print("     class {}".format(sign))
        sign_folder = os.path.join(test_folder, sign)
        for file in os.listdir(sign_folder):
            img_file = os.path.join(test_folder, sign, file)
            img_data.append((file, sign, get_hog(img_file, orientations=bins, pix_per_cell=pix_per_cell)))
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


def get_hog(im_file, orientations=20, pix_per_cell = (112, 112), cells_per_block=(1,1), visualize=False,
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


def test_hog(astronaut = False, verts = False, good = True, visualize = True,
             cells = (224, 224), orientations = 20, quarters = False):
    if quarters:
        cells = (112, 112)
    if astronaut:
        im = data.astronaut()
    elif verts:
        im = os.path.join(os.getcwd(), "Image_features", "vertical_lines_test.jpg")
    elif good:
        im = os.path.join(os.getcwd(), "Train", "27", "00027_00002_00029.png")
    else:
        im = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
    a = get_hog(im, visualize=visualize, pix_per_cell=cells, orientations=orientations)
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


def debug_dataset():
    # gets the hog data for small test dataset:
    f = store_csv_hog(img_folder=os.path.join(os.getcwd(), "Debug"), validation=False)
    # cluster the hog data for k value of 2, 3, 4 and 5:
    hog_kmeans_linear(2, 5, f)
    shapes_in_clusters(f)
    # folder_name = f.split(".")[0]
    # load_all_kmeans(folder_name)
    #use_models_to_predict_new("80hog_img_features_small_test_dataset_2021-03-24.csv", f)


def calculate_cluster_impurity(root_folder_name = None, cluster = 6, verbose = False, normalize = True,
                               purity = True, class_distribution = True):
    # normalize - divide raw entropies by the sum of all entropies so they all sum to 1
    # purity - return purity (1 - impurity)

    # returns a tuple of purities (dict of purities where keys are the clusters and values are the purities of that
    # cluster), class balance for each cluster (dict where the keys represent cluster names, and values are a dict
    # where the keys are the class names and the values are the % makeup of that cluster by that class)
    # class_distribution is true, then the second element of the tuple is a class balance for each cluster, dict
    # where the keys are the class names and the values are the % of all class images in that cluster

    clusters = [cluster]
    if root_folder_name is None:
        root_folder_name = "80hog_img_features_GTSRB_ResNet_2021-03-11"
    csv_name = root_folder_name + ".csv"
    cluster_dist, class_dist = classes_in_clusters(csv_name, exclusive_clusters=clusters, save=False)
    x = cluster_dist
    if class_distribution:
        x = {}
        for class_name in class_dist:
            for cluster_name in class_dist[class_name]:
                if cluster_name not in x:
                    x[cluster_name] = {}
                x[cluster_name][class_name] = class_dist[class_name][cluster_name]
    if verbose:
        print(x)
    entropies = {}
    for cluster in x:
        entropy = 0
        for classname in x[cluster]:
            prob = x[cluster][classname]
            if prob == 0:
                continue
            entropy += -1 * prob * math.log(prob)
        entropies[cluster] = entropy

    if normalize:
        sum = 0
        for i in entropies:
            sum += entropies[i]
        for i in entropies:
            entropies[i] /= sum

    if purity:
        for i in entropies:
            entropies[i] = 1 - entropies[i]

    if verbose:
        print(entropies)
    return entropies, x


def k_value_w_highest_purity(min_k = 2, max_k = 44):
    # calculates the average impurity of the clusters for each value of k from min_k to max_k, and returns the k value
    # with the lowest average impurity
    lowest_impurity = float('inf')
    best_k = 0
    for k in range(min_k, max_k):
        entropies = list(calculate_cluster_impurity(k).values())
        avg_impurity = sum(entropies)/len(entropies)
        print("K value: {}\tAvg impurity: {}".format(k, avg_impurity))
        if avg_impurity < lowest_impurity:
            best_k = k
    print("Best k value is: {}".format(best_k))
    return best_k


def plot_sil_scores(root_folder_name, sil_scores = None, save = False, show = False):
    # save: save the plot
    # show: show the plot

    # if sil_scores is none:
    # uses the sil_scores.txt file in the folder cwd/Image_features/HOG/<root_folder_name>/cluster_metrics
    # to plot the siloutte scores by k value

    #else: saves the plotted sil_scores as a .png and saves the sil scores as a .txt

    sil_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                            "cluster_metrics", "sil_scores.txt")

    if sil_scores is None:
        sil_scores = np.loadtxt(sil_file)
    k_values = [x for x in range(2, len(sil_scores)+2)]
    plt.plot(k_values, sil_scores)
    plt.title("Silhouette Score for different values of k,\n{} dataset".format(root_folder_name))
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette Score")
    if save:
        cluster_metrics_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                              "cluster_metrics")
        if not os.path.exists(cluster_metrics_folder):
            os.mkdir(cluster_metrics_folder)

        figname = "sil_scores.png"
        plt.savefig(os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                 cluster_metrics_folder, figname))

        # -----------save as txt file-----------
        sil_scores = np.array(sil_scores)
        sil_name = "sil_scores"
        sil_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                cluster_metrics_folder, sil_name + ".txt")
        while os.path.exists(sil_file):
            sil_name += "_"
            sil_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                    cluster_metrics_folder, sil_name + ".txt")
        np.savetxt(sil_file, sil_scores)
    if show:
        plt.show()


def plot_cost_scores(root_folder_name, cost_scores=None, save=False, show=False, recommended = True):
    # cost is the sum of squared distances of samples to their closest cluster center.
    # save: save the plot
    # show: show the plot
    # recommended: calculate and print recommended elbow point

    # if cost_scores is none:
    # uses the cost_scores.txt file in the folder cwd/Image_features/HOG/<root_folder_name>/cluster_metrics
    # to plot the cost scores by k value

    # else: saves the plotted cost_scores as a .png and saves the cost scores as a .txt

    cost_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                             "cluster_metrics", "cost_scores.txt")

    if cost_scores is None:
        cost_scores = np.loadtxt(cost_file)
    k_values = [x for x in range(2, len(cost_scores) + 2)]
    plt.plot(k_values, cost_scores)
    plt.title("Inertia Score for different values of k,\n{} dataset".format(root_folder_name))
    plt.xlabel("Number of clusters k")
    plt.ylabel("sum of squared distances of samples\nto their closest cluster center.")
    if save:
        cluster_metrics_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                              "cluster_metrics")
        if not os.path.exists(cluster_metrics_folder):
            os.mkdir(cluster_metrics_folder)

        figname = "cost_scores.png"
        plt.savefig(os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                 cluster_metrics_folder, figname))

        # -----------save as txt file-----------
        cost_scores = np.array(cost_scores)
        cost_name = "cost_scores"
        cost_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name, cluster_metrics_folder,
                                 cost_name + ".txt")
        while os.path.exists(cost_file):
            cost_name += "_"
            cost_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                     cluster_metrics_folder, cost_name + ".txt")
        np.savetxt(cost_file, cost_scores)
    if show:
        plt.show()
    if recommended:
        # code adapted from
        # https://datascience.stackexchange.com/questions/57122/
        # in-elbow-curve-how-to-find-the-point-from-where-the-curve-starts-to-rise
        data = [[k_values[i], cost_scores[i]] for i in range(len(cost_scores))]
        data = np.array(data)

        scaler = MinMaxScaler(feature_range=(0, max(k_values)))
        data = scaler.fit_transform(data)

        rotor = Rotor()
        rotor.fit_rotate(data)
        elbow_index = rotor.get_elbow_index() + 2
        print("Recommended elbow is {} clusters".format(elbow_index))


def plot_davies_bouldin_scores(root_folder_name, db_scores=None, save=False, show=False, recommended = True):
    # save: save the plot
    # show: show the plot
    # recommended: calculate and print recommended cluster size

    # if db_scores is none:
    # uses the db_scores.txt file in the folder cwd/Image_features/HOG/<root_folder_name>/cluster_metrics
    # to plot the db scores by k value

    # else: saves the plotted db_scores as a .png and saves the db scores as a .txt

    db_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                           "cluster_metrics", "davies_bouldin_scores.txt")

    if db_scores is None:
        db_scores = np.loadtxt(db_file)
    k_values = [x for x in range(2, len(db_scores) + 2)]
    plt.plot(k_values, db_scores)
    plt.title("DB Score for different values of k,\n{} dataset".format(root_folder_name))
    plt.xlabel("Number of clusters k")
    plt.ylabel("Davies Bouldin Score (lower is better)")
    if save:
        cluster_metrics_folder = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                              "cluster_metrics")
        if not os.path.exists(cluster_metrics_folder):
            os.mkdir(cluster_metrics_folder)

        figname = "davies_bouldin_scores.png"
        plt.savefig(os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                 cluster_metrics_folder, figname))

        # -----------save as txt file-----------
        db_scores = np.array(db_scores)
        db_name = "db_scores"
        db_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name, cluster_metrics_folder,
                                 db_name + ".txt")
        while os.path.exists(db_file):
            db_name += "_"
            db_file = os.path.join(os.getcwd(), "Image_features", "HOG", root_folder_name,
                                     cluster_metrics_folder, db_name + ".txt")
        np.savetxt(db_file, db_scores)
    if show:
        plt.show()
    if recommended:
        db_scores = list(db_scores)
        best_k = db_scores.index(min(db_scores)) + 2
        print("Recommended k value is {} clusters".format(best_k))



def calculate_costs_inertia_from_models(root_folder_name):
    # called when you want to get the costs from saved kmeans models and save the costs as a txt file without
    # redoing the clustering

    # array of tuples (k value, inertia for that k value)  This dict will later get converted to a list
    # of intertia values sorted by k value
    inertia = []

    # load the models
    models_and_centers = load_all_kmeans(root_folder_name)
    for model, centers in models_and_centers:
        num_clusters = len(centers)
        inertia.append((num_clusters, model.inertia_))
        print("Calculated inertia for k value of {}".format(num_clusters))
    inertia = sorted(inertia, key=lambda x: x[0])
    inertia = [x[1] for x in inertia]
    plot_cost_scores(root_folder_name, cost_scores=inertia, save=True)


def calculate_siloutte_from_models(root_folder_name, save = True, show = False, limit = None):
    # called when you want to get the siloutte scores from saved kmeans models and save the them as a txt file without
    # redoing the clustering

    # save and true are parameters to pass to plot_sil_scores()
    # limit (int if not None) will only do the first <limit> values of k

    filename = root_folder_name + ".csv"
    bins = ""
    for i in filename:
        if i.isdigit():
            bins += i
        else:
            break
    bins = int(bins)
    cols = [x for x in range(2, bins + 2, 1)]
    df = load_hog_df(filename, usecols=cols)
    a = df.values

    # array of tuples (k value, sil score for that k value)  This dict will later get converted to a list
    # of sil_score values sorted by k value
    sil_scores = []

    # counter for if limit parameter is used
    count = 0

    # load the models
    models_and_centers = load_all_kmeans(root_folder_name)
    for model, centers in models_and_centers:
        if limit is not None and count == limit:
            break
        num_clusters = len(centers)
        labels = model.predict(a)
        sil_score = silhouette_score(a, labels, metric='euclidean')
        sil_scores.append((num_clusters, sil_score))
        print("Calculated sil score for k = {}, {}".format(num_clusters, sil_score))
        count += 1
    sil_scores = sorted(sil_scores, key=lambda x: x[0])
    sil_scores = [x[1] for x in sil_scores]
    plot_sil_scores(root_folder_name, sil_scores=sil_scores, save=save, show = show)


def calculate_davies_bouldin_from_models(root_folder_name, limit = None):
    # called when you want to get the davies bouldin scores from saved kmeans models and save as a txt file without
    # redoing the clustering

    # array of tuples (k value, db score for that k value)  This dict will later get converted to a list
    # of intertia values sorted by k value
    db_scores = []

    filename = root_folder_name + ".csv"
    bins = ""
    for i in filename:
        if i.isdigit():
            bins += i
        else:
            break
    bins = int(bins)
    cols = [x for x in range(2, bins + 2, 1)]
    df = load_hog_df(filename, usecols=cols)
    a = df.values

    # array of tuples (k value, db score for that k value)  This dict will later get converted to a list
    # of db_scores values sorted by k value
    db_scores = []

    # counter for if limit parameter is used
    count = 0

    # load the models
    models_and_centers = load_all_kmeans(root_folder_name)
    for model, centers in models_and_centers:
        if limit is not None and count == limit:
            break
        num_clusters = len(centers)
        labels = model.predict(a)
        db_score = davies_bouldin_score(a, labels)
        db_scores.append((num_clusters, db_score))
        print("Calculated db score for k = {}, db score: {}".format(num_clusters, db_score))
        count += 1
    db_scores = sorted(db_scores, key=lambda x: x[0])
    db_scores = [x[1] for x in db_scores]
    plot_davies_bouldin_scores(root_folder_name, db_scores, save=True, show=True)

if __name__ == '__main__':
    main_folder_name = "80hog_img_features_GTSRB_ResNet_2021-03-11"
    main_csv_name = main_folder_name + ".csv"
    test_folder = "80hog_img_features_Test_2021-03-22"
    # a = calculate_cluster_impurity(main_folder_name, cluster=3, verbose=True)
    # print(a)
    #k_value_w_highest_purity()
    #plot_sil_scores(main_folder_name)
    #calculate_costs_intertia_from_models(main_folder_name)
    #classes_in_clusters(main_csv_name)
    #calculate_davies_bouldin_from_models(main_folder_name)
    #unsupervised_predict_all(root_folder_name=main_folder_name, k_value=3)
    #unsupervised_predict_all(root_folder_name=main_folder_name, k_value=3,
    #                         predict_folder_name=test_folder)
    test_all_cluster_pred_vector(root_folder_name=main_folder_name, pred_folder_name=main_folder_name)