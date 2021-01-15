from tree_classes import *
from PIL import Image
import numpy as np
from general import *
from tree_helper import signs, print_tree, Tree_stats, split_all, split_signs
from pytorch_resnet import create_and_train_model, load_model, test_model_manually, \
    test_attribute_model_manually, test_final_classifier_manually, preprocess_image, \
    test_final_classifier_manually_byclass
import os
from shutil import copyfile

def shape_mapping():
    return {"circle": 0, "diamond" : 1, "inverted_triangle": 2, "octagon": 3, "triangle": 4}

def circle_signs():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

def triangle_signs():
    return [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

def red_circular_signs():
    return [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]

def white_circular_signs():
    return [6, 32, 41, 42]

def blue_circular_signs():
    return [33, 34, 35, 36, 37, 38, 39, 40]

def triangular_road_true_signs():
    return [11, 19, 20, 21, 24]

def triangular_road_false_signs():
    return [18, 22, 23, 25, 26, 27, 28, 29, 30, 31]


def complement_signs(signs_array):
    # takes the complement of a set of signs <signs_array> which is an array of ints
    all_signs = [i for i in range(43)]
    return list(set(all_signs) - set(signs_array))


class nndt_depth3_unweighted(tree):
    def __init__(self):
        # creates the structure of the tree
        root = classifier("root", "shape")
        model_folder = os.path.join(os.getcwd(), "nndt_data", "nndt3_unweighted")
        root.neuralnet = load_model(os.path.join(model_folder, "shape", "shape_resnet_2021-01-04"))
        root.neuralnet.eval()
        self.rootnode = root
        super().__init__(root)
        signsarray = signs()

        # add signs 12, 13, and 14
        root.add_child(signsarray[12])
        signsarray[12].parent = root
        self.add_node(signsarray[12])

        root.add_child(signsarray[13])
        signsarray[13].parent = root
        self.add_node(signsarray[13])

        root.add_child(signsarray[14])
        signsarray[14].parent = root
        self.add_node(signsarray[14])

        # add final classifiers and none nodes
        circle_classifier = final_classifier("circle", root)
        circle_classifier.neuralnet = load_model(os.path.join(model_folder, "circle_final_classifier",
                              "circle_final_classifier_resnet_2021-01-05"))
        circle_classifier.neuralnet.eval()
        root.add_child(circle_classifier)
        self.add_node(circle_classifier)

        circle_none = node("circle_none", circle_classifier)
        circle_classifier.add_child(circle_none)
        self.add_node(circle_none)

        triangle_classifier = final_classifier("triangle", root)
        triangle_classifier.neuralnet = load_model(os.path.join(model_folder, "triangle_final_classifier",
                              "triangle_final_classifier_resnet_2021-01-11"))
        triangle_classifier.neuralnet.eval()
        root.add_child(triangle_classifier)
        self.add_node(triangle_classifier)

        triangle_none = node("triangle_none", triangle_classifier)
        triangle_classifier.add_child(triangle_none)
        self.add_node(triangle_none)

        # add leaf nodes under final classifiers
        for sign in signsarray:
            if sign.properties["shape"] == "triangle":
                triangle_classifier.add_child(sign)
                sign.parent = triangle_classifier
                self.add_node(sign)
            if sign.properties["shape"] == "circle":
                circle_classifier.add_child(sign)
                sign.parent = circle_classifier
                self.add_node(sign)

        # fill out the .pred_value_names parameter of the classifiers and final classifiers
        root.pred_value_names = [circle_classifier, signsarray[12], signsarray[13], signsarray[14], triangle_classifier]
        triangle_classifier.pred_value_names = [signsarray[x] for x in triangle_signs()]
        triangle_classifier.pred_value_names.append(triangle_none)
        circle_classifier.pred_value_names = [signsarray[x] for x in circle_signs()]
        circle_classifier.pred_value_names.append(circle_none)

    def prediction_vector(self, image, dict = True, path = True):
        # path (bool) indicates whether <image> is a path to an image or a tensor representing an image
        # image (tensor or string) image in tensor form or full path to an image
        # dict indicates whether or not to return a dictionary or array
        # unnormalized vector, probability taken away by none nodes
        if (isinstance(image, np.ndarray)):
            image = Image.fromarray(image)
        image = preprocess_image(image, path = path)
        pred = self.root.predict(1, image)

        # normalize the vector and format as 2 digits
        total_prob = 0
        for key in pred:
            total_prob += pred[key]
        if dict:
            # make a dictionary to return
            normalized = {}
            for key in pred:
                normalized[format_two_digits(key)] = pred[key]/total_prob
            return normalized
        else:
            # make an array to return
            normalized = [x for x in range(43)]
            for key in pred:
                normalized[int(key)] = pred[key]/total_prob
            return normalized

    def prediction(self, pred_vector):
        # pred_vector is a dict from prediction_vector()
        max_val = 0
        pred_class = None
        for key in pred_vector:
            if pred_vector[key] > max_val:
                max_val = pred_vector[key]
                pred_class = key
        return pred_class, max_val

    def generate_all_final_classifier_datasets(self):
        # there are 2 final classifiers, circle and triangle
        # circle:
        circle_signs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        circle_folder = "circle_final_classifier"
        generate_attribute_dataset_final_classifier(circle_signs, circle_folder, "nndt3_unweighted")

        # triangle:
        triangle_signs = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        triangle_folder = "triangle_final_classifier"
        generate_attribute_dataset_final_classifier(triangle_signs, triangle_folder, "nndt3_unweighted")

    @staticmethod
    def leaf_classifier_groups():
        # returns a 2d array for untargeted attack over multiple classifiers
        # 1st dimension represents the leaf classifiers for this nndt (circular signs, triangular signs, shape)
        # and 2nd dimension represents the classs classified by the leaf classifier
        return [circle_signs(), triangle_signs(), [12, 13, 14]]

    def test(self, path = None, verbose=True, limit=None, startlimit = None, exclusive = None):
        test_model_manually(self, verbose=verbose, nndt=True, limit=limit, startlimit=startlimit,
                            exclusive=exclusive, path=path)


class nndt_depth4_unweighted(tree):
    def __init__(self):
        # creates the structure of the tree
        root = classifier("root", "shape")
        model_folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted")
        # same root neuralnet as nndt3_unweighted
        root.neuralnet = load_model(os.path.join(os.getcwd(), "nndt_data", "nndt3_unweighted", "shape",
                              "shape_resnet_2021-01-04"))
        root.neuralnet.eval()
        self.rootnode = root
        super().__init__(root)
        signsarray = signs()

        # add signs 12, 13, and 14
        root.add_child(signsarray[12])
        signsarray[12].parent = root
        self.add_node(signsarray[12])

        root.add_child(signsarray[13])
        signsarray[13].parent = root
        self.add_node(signsarray[13])

        root.add_child(signsarray[14])
        signsarray[14].parent = root
        self.add_node(signsarray[14])

        # add circle_color classifier and none nodes, and all classifiers below it
        circle_color_classifier = classifier("circle", "color", root)
        # TODO
        circle_color_classifier.neuralnet = load_model(os.path.join(model_folder, "circle_final_classifier",
                              "circle_final_classifier_resnet_2021-01-05"))
        circle_color_classifier.neuralnet.eval()
        root.add_child(circle_color_classifier)
        self.add_node(circle_color_classifier)

        circle_color_none = node("circle_color_none", circle_color_classifier)
        circle_color_classifier.add_child(circle_color_none)
        self.add_node(circle_color_none)

        # red circular signs
        red_circular_classifier = final_classifier("red_circular", circle_color_classifier)
        # TODO
        red_circular_classifier.neuralnet = load_model(os.path.join(model_folder, "nndt_data"))
        red_circular_classifier.neuralnet.eval()
        circle_color_classifier.add_child(red_circular_classifier)
        self.add_node(red_circular_classifier)

        red_circular_none = node("red_circular_none", red_circular_classifier)
        red_circular_classifier.add_child(red_circular_none)
        self.add_node(red_circular_none)

        # blue circular signs
        blue_circular_classifier = final_classifier("blue_circular", circle_color_classifier)
        # TODO
        blue_circular_classifier.neuralnet = load_model(os.path.join(model_folder, "nndt_data"))
        blue_circular_classifier.neuralnet.eval()
        circle_color_classifier.add_child(blue_circular_classifier)
        self.add_node(blue_circular_classifier)

        blue_circular_none = node("blue_circular_none", blue_circular_classifier)
        blue_circular_classifier.add_child(blue_circular_none)
        self.add_node(blue_circular_none)

        # white circular signs
        white_circular_classifier = final_classifier("white_circular", circle_color_classifier)
        # TODO
        white_circular_classifier.neuralnet = load_model(os.path.join(model_folder, "nndt_data"))
        white_circular_classifier.neuralnet.eval()
        circle_color_classifier.add_child(white_circular_classifier)
        self.add_node(white_circular_classifier)

        white_circular_none = node("white_circular_none", white_circular_classifier)
        white_circular_classifier.add_child(white_circular_none)
        self.add_node(white_circular_none)

        # add triangle_road classifier and none nodes, and all nodes below it
        triangle_road_classifier = classifier("triangle", "road", root)
        # TODO
        triangle_road_classifier.neuralnet = load_model(os.path.join(model_folder, "triangle_final_classifier",
                              "triangle_final_classifier_resnet_2021-01-05"))
        triangle_road_classifier.neuralnet.eval()
        root.add_child(triangle_road_classifier)
        self.add_node(triangle_road_classifier)

        triangle_road_none = node("triangle_road_none", triangle_road_classifier)
        triangle_road_classifier.add_child(triangle_road_none)
        self.add_node(triangle_road_none)

        # triangle signs with road
        triangle_road_true_classifier = final_classifier("triangle_road_true", triangle_road_classifier)
        # TODO
        triangle_road_true_classifier.neuralnet = load_model(os.path.join(model_folder, "nndt_data"))
        triangle_road_true_classifier.neuralnet.eval()
        triangle_road_classifier.add_child(triangle_road_true_classifier)
        self.add_node(triangle_road_true_classifier)

        triangle_road_true_none = node("triangle_road_true_none", triangle_road_true_classifier)
        triangle_road_true_classifier.add_child(triangle_road_true_none)
        self.add_node(triangle_road_true_none)

        # triangle signs without road
        triangle_road_false_classifier = final_classifier("triangle_road_false", triangle_road_classifier)
        # TODO
        triangle_road_false_classifier.neuralnet = load_model(os.path.join(model_folder, "nndt_data"))
        triangle_road_false_classifier.neuralnet.eval()
        triangle_road_classifier.add_child(triangle_road_false_classifier)
        self.add_node(triangle_road_false_classifier)

        triangle_road_false_none = node("triangle_road_false_none", triangle_road_false_classifier)
        triangle_road_false_classifier.add_child(triangle_road_false_none)
        self.add_node(triangle_road_false_none)

        # add leaf nodes under final classifiers
        for sign in signsarray:
            if sign.properties["shape"] == "circle":
                if sign.properties["color"] == "red":
                    red_circular_classifier.add_child(sign)
                    sign.parent = red_circular_classifier
                    self.add_node(sign)
                if sign.properties["color"] == "blue":
                    blue_circular_classifier.add_child(sign)
                    sign.parent = blue_circular_classifier
                    self.add_node(sign)
                if sign.properties["color"] == "white":
                    white_circular_classifier.add_child(sign)
                    sign.parent = white_circular_classifier
                    self.add_node(sign)
            if sign.properties["shape"] == "triangle":
                if sign.properties["road"] == True:
                    triangle_road_true_classifier.add_child(sign)
                    sign.parent = triangle_road_true_classifier
                    self.add_node(sign)
                if sign.properties["road"] == False:
                    triangle_road_false_classifier.add_child(sign)
                    sign.parent = triangle_road_false_classifier
                    self.add_node(sign)

        # fill out the .pred_value_names parameter of the classifiers and final classifiers
        # TODO
        # root.pred_value_names = [circle_classifier, signsarray[12], signsarray[13], signsarray[14], triangle_classifier]
        # triangle_classifier.pred_value_names = [signsarray[x] for x in triangle_signs()]
        # triangle_classifier.pred_value_names.append(triangle_none)
        # circle_classifier.pred_value_names = [signsarray[x] for x in circle_signs()]
        # circle_classifier.pred_value_names.append(circle_none)

    def prediction_vector(self, image, dict = True, path = True):
        # path (bool) indicates whether <image> is a path to an image or a tensor representing an image
        # image (tensor or string) image in tensor form or full path to an image
        # dict indicates whether or not to return a dictionary or array
        # unnormalized vector, probability taken away by none nodes
        if (isinstance(image, np.ndarray)):
            image = Image.fromarray(image)
        image = preprocess_image(image, path = path)
        pred = self.root.predict(1, image)

        # normalize the vector and format as 2 digits
        total_prob = 0
        for key in pred:
            total_prob += pred[key]
        if dict:
            # make a dictionary to return
            normalized = {}
            for key in pred:
                normalized[format_two_digits(key)] = pred[key]/total_prob
            return normalized
        else:
            # make an array to return
            normalized = [x for x in range(43)]
            for key in pred:
                normalized[int(key)] = pred[key]/total_prob
            return normalized

    def prediction(self, pred_vector):
        # pred_vector is a dict from prediction_vector()
        max_val = 0
        pred_class = None
        for key in pred_vector:
            if pred_vector[key] > max_val:
                max_val = pred_vector[key]
                pred_class = key
        return pred_class, max_val

    @staticmethod
    def generate_all_final_classifier_datasets():
        classname = "nndt4_unweighted"
        # there are 3 circular final classifiers, red, white, and blue
        # red:
        red_signs = red_circular_signs()
        red_folder = "red_circular_final_classifier"
        generate_attribute_dataset_final_classifier(red_signs, red_folder, classname)

        # white:
        white_signs = white_circular_signs()
        white_folder = "white_circular_final_classifier"
        generate_attribute_dataset_final_classifier(white_signs, white_folder, classname)

        # blue:
        blue_signs = blue_circular_signs()
        blue_folder = "blue_circular_final_classifier"
        generate_attribute_dataset_final_classifier(blue_signs, blue_folder, classname)

        # there are 2 triangular final classifiers, road_true and road_false:
        road_true_signs = triangular_road_true_signs()
        road_true_folder = "triangular_road_true_final_classifier"
        generate_attribute_dataset_final_classifier(road_true_signs, road_true_folder, classname)

        road_false_signs = triangular_road_false_signs()
        road_false_folder = "triangular_road_false_final_classifier"
        generate_attribute_dataset_final_classifier(road_false_signs, road_false_folder, classname)

    @staticmethod
    def generate_all_classifier_datasets():
        # the root classifier of shape is the same as for nndt3_unweighted, so no need to recreate it
        # there are 2 other classifiers: circular signs which are classified based on color and triangular signs
        # which are classified based on road

        # circular signs classified by color:
        generate_classifier_dataset(circle_signs(), "color", "circle_color", "nndt4_unweighted")

        # triangular signs classified by road
        generate_classifier_dataset(triangle_signs(), "road", "triangle_road", "nndt4_unweighted")

    def test(self, path = None, verbose=True, limit=None, startlimit = None, exclusive = None):
        test_model_manually(self, verbose=verbose, nndt=True, limit=limit, startlimit=startlimit,
                            exclusive=exclusive, path=path)

    @staticmethod
    def test_sign_set():
        a = []
        a.extend(triangular_road_false_signs())
        a.extend(triangular_road_true_signs())
        a.extend(red_circular_signs())
        a.extend(blue_circular_signs())
        a.extend(white_circular_signs())
        a.extend([12, 13, 14])
        a.sort()
        b = [i for i in range(43)]
        print(a==b)

    @staticmethod
    def test_classifiers(byclass = False, exclusive = False, testfolder = None, save = False, verbose = False):
        if testfolder == None:
            testfolder = os.path.join(os.getcwd(), "Test")
        circle_mapping = ["red", "white", "blue", "none"]
        triangle_mapping = ["False", "none", "True"]
        classifier_dict = {"circle_color": ("color", circle_signs(), circle_mapping),
                           "triangle_road": ("road", triangle_signs(), triangle_mapping)}
        for classifier in classifier_dict:
            folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted", classifier)
            model_file = os.path.join(folder, classifier + "_resnet_2021-01-13")
            model = load_model(model_file)
            attribute, correct_signs, mapping = classifier_dict[classifier]
            exc = correct_signs if exclusive else None
            # make filename
            filename = None
            if save:
                fname = "exclusive_" if exclusive else ""
                fname += os.path.split(testfolder)[1].lower() + "_acc"
                fname += "_byclass" if byclass else ""
                fname += ".txt"
                filename = os.path.join(folder, fname)
            test_attribute_model_manually(model, attribute, correct_signs, mapping, path=testfolder, byclass = byclass,
                                          exclusive = exc, save_file=filename, verbose = verbose)


    @staticmethod
    def test_final_classifiers(byclass = False, exclusive = False, testfolder = None, save = False, verbose = False,
                               top_misclassifications = None):
        if testfolder == None:
            testfolder = os.path.join(os.getcwd(), "Test")
        fc_dict = {"blue_circle": blue_circular_signs(),
                   "red_circle": red_circular_signs(),
                   "white_circle": white_circular_signs(),
                   "triangular_road_false": triangular_road_false_signs(),
                   "triangular_road_true": triangular_road_true_signs()}
        for classifier in fc_dict:
            folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted", classifier)
            model_file = os.path.join(folder, classifier + "_final_classifier_resnet_2021-01-13")
            model = load_model(model_file)
            correct_signs = fc_dict[classifier]
            exc = correct_signs if exclusive else None
            # make filename
            filename = None
            if save:
                fname = "exclusive_" if exclusive else ""
                fname += os.path.split(testfolder)[1].lower() + "_acc"
                fname += "_byclass" if byclass else ""
                fname += ".txt"
                filename = os.path.join(folder, fname)
            if not byclass:
                test_final_classifier_manually(model, correct_signs, path = testfolder, verbose=verbose, exclusive=exc,
                                               save_file = filename)
            else:
                test_final_classifier_manually_byclass(model, correct_signs, path=testfolder, verbose=verbose,
                                                       exclusive=exc, top_misclassifications = top_misclassifications,
                                                       save_file = filename)


    @staticmethod
    def train_classifiers():
        classifiers_num_classes = {"circle_color": 4, "triangle_road": 3}
        for classifier in classifiers_num_classes:
            create_train_attribute_model(classifier, classifiers_num_classes[classifier], "nndt4_unweighted")

    @staticmethod
    def train_final_classifiers():
        create_train_attribute_model("red_circular_final_classifier", len(red_circular_signs())+1, "nndt4_unweighted")
        create_train_attribute_model("white_circular_final_classifier", len(white_circular_signs()) + 1, "nndt4_unweighted")
        create_train_attribute_model("blue_circular_final_classifier", len(blue_circular_signs()) + 1, "nndt4_unweighted")
        create_train_attribute_model("triangular_road_true_final_classifier", len(triangular_road_true_signs()) + 1,
                                     "nndt4_unweighted")
        create_train_attribute_model("triangular_road_false_final_classifier", len(triangular_road_false_signs()) + 1,
                                     "nndt4_unweighted")


def test_train_val_folders(folder_root):
    attribute_test_folder = os.path.join(folder_root, "Test")
    os.mkdir(attribute_test_folder)
    attribute_train_folder = os.path.join(folder_root, "Train")
    os.mkdir(attribute_train_folder)
    attribute_val_folder = os.path.join(folder_root, "Validation")
    os.mkdir(attribute_val_folder)
    subfolders = {"Validation": attribute_val_folder, "Test": attribute_test_folder, "Train": attribute_train_folder}
    return subfolders


def generate_classifier_dataset(road_signs, attribute, filename, classinstance):
    # takes in a list of road signs that will reach this classifier and reformats the GTSRB dataset into Test, Train,
    # and Validation folders according to <attribute>.  Each of these folders will have subfolders corresponding to
    # the classes resulting from splitting the dataset based on <attribute>.  The reformatted dataset will be stored
    # in the ./nndt_data/<classinstance> folder under a root folder named <attribute>.
    # for a root node of an nndt, road signs will be all classes: [0, 1, 2, ..., 42, 43]

    # make root folder and test, train, val folders
    root_folder = os.path.join(os.getcwd(), "nndt_data", classinstance, filename)
    os.mkdir(root_folder)
    subfolders = test_train_val_folders(root_folder)

    # split up signs into groups
    signarray = signs()
    included = [signarray[i] for i in road_signs]
    values, _ = split_signs(included, attribute)

    # road_signs are all the classes classified by the classifier
    road_signs = [format_two_digits(x) for x in road_signs]

    # key: Test, Train, or Validation
    for key in subfolders:
        print("Working on folder " + key + "\n\n")
        subfolder_root = subfolders[key]
        original_folder = os.path.join(os.getcwd(), key)
        # make subfolders based on the attribute classes
        for value in values:
            os.mkdir(os.path.join(subfolder_root, str(value)))

        # make none class
        os.mkdir(os.path.join(subfolder_root, "none"))

        # iterate through corresponding original dataset folder to copy images into these subfolders
        original_classes = os.listdir(original_folder)
        for i, sign in enumerate(original_classes):
            print("Copying images of class " + sign)

            # if sign is part of the classifier, it's label is it's attribute value, else it is "none"
            sign_value = signarray[i].properties[attribute] if sign in road_signs else "none"
            sign_folder = os.path.join(original_folder, sign)
            for image_file in os.listdir(sign_folder):
                full_img_path = os.path.join(sign_folder, image_file)
                copyfile(full_img_path, os.path.join(subfolder_root, str(sign_value), image_file))


def generate_attribute_dataset_final_classifier(road_signs, filename, classinstance):
    # takes in a list of road signs in the final classifier and reformats the GTSRB dataset into Test, Train,
    # and Validation folders.  Each of these folders will have subfolders corresponding to the classes
    # in <road_signs> and all classes not included in road_signs will be in the none folder. The reformatted
    # dataset will be stored in the ./nndt_data/<classinstance> folder under a root folder named <filename>.

    # make root folder and test, train, val folders
    root_folder = os.path.join(os.getcwd(), "nndt_data", classinstance, filename)
    os.mkdir(root_folder)
    subfolders = test_train_val_folders(root_folder)

    # road_signs are all the classes classified by the final classifier
    road_signs = [format_two_digits(x) for x in road_signs]

    for key in subfolders:
        print("Working on folder " + key + "\n\n")
        subfolder_root = subfolders[key]
        original_folder = os.path.join(os.getcwd(), key)
        # make subfolders based on the attribute classes
        for included_class in road_signs:
            os.mkdir(os.path.join(subfolder_root, included_class))

        # make none class
        os.mkdir(os.path.join(subfolder_root, "none"))

        # iterate through corresponding original dataset folder to copy images into these subfolders
        original_classes = os.listdir(original_folder)
        for sign in original_classes:
            print("Copying images of class " + sign)

            # if sign is part of the final classifier, it's label is it's class name, else it is "none"
            sign_value = sign if sign in road_signs else "none"
            sign_folder = os.path.join(original_folder, sign)
            for image_file in os.listdir(sign_folder):
                full_img_path = os.path.join(sign_folder, image_file)
                copyfile(full_img_path, os.path.join(subfolder_root, sign_value, image_file))


def create_train_attribute_model(attribute, num_classes, classinstance):
    # classinstance is the string version of the nndt class name, specifies the folder in ./nndt_data to look for
    # datasets
    # for classifiers, <attribute> should be the string version of the attribute
    # for final classifiers, <attribute> should be the name of the root folder of the dataset in
    # ./nndt_data/<classinstance>, and make sure to include the none node in the number of classes
    attribute_folder = os.path.join(os.getcwd(), "nndt_data", classinstance, attribute)
    model_filename = attribute + "_resnet_" + str_date()
    model_path = os.path.join(attribute_folder, model_filename)
    create_and_train_model(attribute_folder, model_path, num_classes)


def test_nndt():
    Tree = nndt_depth3_unweighted()
    #Tree_stats(Tree)
    img = os.path.join(os.getcwd(), "Test", "00", "00243.png")
    print(Tree.prediction_vector(img, dict=False))


def test_fc():
    model_file = os.path.join(os.getcwd(), "nndt_data", "nndt3_unweighted", "triangle_final_classifier",
                              "triangle_final_classifier_resnet_2021-01-05")
    model = load_model(model_file)
    test_folder = os.path.join(os.getcwd(), "Test")
    #test_final_classifier_manually(model, triangle_signs(), path = test_folder, verbose=True,
    #                               exclusive=triangle_signs())
    test_final_classifier_manually_byclass(model, triangle_signs(), path=test_folder, verbose=True,
                                           exclusive=triangle_signs()) #, limit=10, top_misclassifications=2)


def test_reg():
    model_file = os.path.join(os.getcwd(), "Models", "pytorch_resnet_saved_11_9_20")
    model = load_model(model_file)
    test_folder = os.path.join(os.getcwd(), "Test")
    test_model_manually(model, test_folder, exclusive=triangle_signs(), byclass=True, top_misclassifications=2)


def create_all_nndt4_unweighted_datasets():
    nndt_depth4_unweighted.generate_all_classifier_datasets()
    nndt_depth4_unweighted.generate_all_final_classifier_datasets()

if __name__ == '__main__':
    # create_train_attribute_model("triangle_road", len(triangle_signs()) + 1, "nndt4_unweighted")
    nndt_depth4_unweighted.train_classifiers()
    print()