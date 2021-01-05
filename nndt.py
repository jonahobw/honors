from tree_classes import *
from general import *
from tree_helper import signs, print_tree, Tree_stats, split_all, split_signs
from pytorch_resnet import create_and_train_model, load_model, test_model_manually, \
    test_attribute_model_manually, test_final_classifier_manually, preprocess_image
import os
from shutil import copyfile

def shape_mapping():
    return {"circle": 0, "diamond" : 1, "inverted_triangle": 2, "octagon": 3, "triangle": 4}

def circle_signs():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

def triangle_signs():
    return [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


class nndt_depth3_unweighted(tree):
    def __init__(self):
        # creates the structure of the tree
        root = classifier("root", "shape")
        root.neuralnet = load_model(os.path.join(os.getcwd(), "nndt_data", "shape",
                              "shape_resnet_2021-01-03"))
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
        circle_classifier.neuralnet = load_model(os.path.join(os.getcwd(), "nndt_data", "circle_final_classifier",
                              "circle_final_classifier_resnet_2021-01-04"))
        root.add_child(circle_classifier)
        self.add_node(circle_classifier)

        circle_none = node("circle_none", circle_classifier)
        circle_classifier.add_child(circle_none)
        self.add_node(circle_none)

        triangle_classifier = final_classifier("triangle", root)
        triangle_classifier.neuralnet = load_model(os.path.join(os.getcwd(), "nndt_data", "triangle_final_classifier",
                              "triangle_final_classifier_resnet_2021-01-04"))
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

    def predict(self, image):
        # input is a full image path
        # todo: normalize and combine?
        return self.root.predict(1, preprocess_image(image))

    def generate_all_final_classifier_datasets(self):
        # there are 2 final classifiers, circle and triangle
        # circle:
        circle_signs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        circle_folder = "circle_final_classifier"
        generate_attribute_dataset_final_classifier(circle_signs, circle_folder)

        # triangle:
        triangle_signs = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        triangle_folder = "triangle_final_classifier"
        generate_attribute_dataset_final_classifier(triangle_signs, triangle_folder)


def generate_attribute_dataset(attribute):
    # takes in an <attribute> (string) and reformats the GTSRB dataset into Test, Train, and Validation folders
    # according to <attribute>.  Each of these folders will have subfolders corresponding to the classes
    # resulting from splitting the dataset based on <attribute>.  The reformatted dataset will be stored in
    # the ./nndt_data folder under a root folder named <attribute>

    # make root folder and test, train, val folders
    attribute_root_folder = os.path.join(os.getcwd(), "nndt_data", attribute)
    os.mkdir(attribute_root_folder)
    attribute_test_folder = os.path.join(attribute_root_folder, "Test")
    os.mkdir(attribute_test_folder)
    attribute_train_folder = os.path.join(attribute_root_folder, "Train")
    os.mkdir(attribute_train_folder)
    attribute_val_folder = os.path.join(attribute_root_folder, "Validation")
    os.mkdir(attribute_val_folder)
    subfolders = {"Validation": attribute_val_folder, "Test": attribute_test_folder, "Train": attribute_train_folder}

    # split up signs into groups
    split = split_all(attribute)
    values, _ = split_signs(signs(), attribute)

    for key in subfolders:
        print("Working on folder " + key + "\n\n")
        subfolder_root = subfolders[key]
        original_folder = os.path.join(os.getcwd(), key)
        # make subfolders based on the attribute classes
        for value in values:
            os.mkdir(os.path.join(subfolder_root, value))

        # iterate through corresponding original dataset folder to copy images into these subfolders
        original_classes = os.listdir(original_folder)
        for sign in original_classes:
            print("Copying images of class " + sign)
            sign_value = split[int(sign)]
            sign_folder = os.path.join(original_folder, sign)
            for image_file in os.listdir(sign_folder):
                full_img_path = os.path.join(sign_folder, image_file)
                copyfile(full_img_path, os.path.join(subfolder_root, sign_value, image_file))


def generate_attribute_dataset_final_classifier(road_signs, filename):
    # takes in a list of road signs in the final classifier and reformats the GTSRB dataset into Test, Train,
    # and Validation folders.  Each of these folders will have subfolders corresponding to the classes
    # in <road_signs> and all classes not included in road_signs will be in the none folder. The reformatted
    # dataset will be stored in the ./nndt_data folder under a root folder named <filename>.

    # make root folder and test, train, val folders
    attribute_root_folder = os.path.join(os.getcwd(), "nndt_data", filename)
    os.mkdir(attribute_root_folder)
    attribute_test_folder = os.path.join(attribute_root_folder, "Test")
    os.mkdir(attribute_test_folder)
    attribute_train_folder = os.path.join(attribute_root_folder, "Train")
    os.mkdir(attribute_train_folder)
    attribute_val_folder = os.path.join(attribute_root_folder, "Validation")
    os.mkdir(attribute_val_folder)
    subfolders = {"Validation": attribute_val_folder, "Test": attribute_test_folder, "Train": attribute_train_folder}

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


def create_train_attribute_model(attribute, num_classes):
    # for classifiers, <attribute> should be the string version of the attribute
    # for final classifiers, <attribute> should be the name of the root folder of the dataset in ./nndt_data
    attribute_folder = os.path.join(os.getcwd(), "nndt_data", attribute)
    model_filename = attribute + "_resnet_" + str_date()
    model_path = os.path.join(attribute_folder, model_filename)
    create_and_train_model(attribute_folder, model_path, num_classes)

def test_nndt():
    Tree = nndt_depth3_unweighted()
    Tree_stats(Tree)

def test_data():
    model_file = os.path.join(os.getcwd(), "nndt_data", "circle_final_classifier",
                              "circle_final_classifier_resnet_2021-01-04")
    model = load_model(model_file)
    test_folder = os.path.join(os.getcwd(), "Test")
    test_final_classifier_manually(model, circle_signs(), path=test_folder, verbose=True, limit=1)

# generate_attribute_dataset("shape")
create_train_attribute_model("shape", 5)
#
# img_file = os.path.join(os.getcwd(), "Test", "00", "00243.png")
# nndt = nndt_depth3_unweighted()
# a = nndt.predict(img_file)
# print(a)