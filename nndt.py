from tree_classes import *
from general import *
from tree_helper import signs, print_tree, Tree_stats, split_all, split_signs
from pytorch_resnet import create_and_train_model
import os
from shutil import copyfile

class nndt_depth3_unweighted(tree):
    def __init__(self):
        # creates the structure of the tree
        # todo fill out the .neuralnet parameter of the classifiers and final classifiers
        # todo fill out the .pred_value_names parameter of the classifiers and final classifiers
        root = classifier("root", "shape")
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
        root.add_child(circle_classifier)
        self.add_node(circle_classifier)

        circle_none = node("circle_none", circle_classifier)
        circle_classifier.add_child(circle_none)
        self.add_node(circle_none)

        triangle_classifier = final_classifier("triangle", root)
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

    def predict(self, image):
        # todo: normalize and combine?
        return self.root.predict(1, image)


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


def create_train_attribute_model(attribute):
    attribute_folder = os.path.join(os.getcwd(), "nndt_data", attribute)
    model_filename = attribute + "_resnet_" + str_date()
    model_path = os.path.join(attribute_folder, model_filename)
    create_and_train_model(attribute_folder, model_path)

def test_nndt():
    Tree = nndt_depth3_unweighted()
    Tree_stats(Tree)

# generate_attribute_dataset("shape")
# create_train_attribute_model("shape")