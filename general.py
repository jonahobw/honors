import datetime
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

def str_date():
    # returns the date as a string formatted as <year>-<month>-<day>
    a = str(datetime.datetime.now())
    b = a.split(" ")[0]
    return b

def format_two_digits(number):
    # parameters:
    # number (int, float, or string): number to be converted
    #
    # return values:
    # two_digits (string): the input number converted to a string and padded
    # with zeros on the front so that it is at least 2 digits long
    return str(number).zfill(2)

def count_folder_contents(root_folder = None, verbose = False):
    if root_folder == None:
        root_folder = os.path.join(os.getcwd())
    total_count = 0
    for folder in ["Train", "Validation", "Test"]:
        print(folder + " Folder")
        folder_count = 0
        folder_root = os.path.join(root_folder, folder)
        subfolders = [os.path.join(folder_root, x) for x in os.listdir(folder_root)]
        for subfolder in subfolders:
            class_count = len(os.listdir(subfolder))
            folder_count += class_count
            if verbose:
                sign_class = os.path.split(subfolder)[1]
                print("Class " + sign_class + ": " + str(class_count) + " images")
        print(str(folder_count) + " images total\n")
        total_count += folder_count
    print("Total number of images: " + str(total_count))


def num_images(folder = None, show_output = True):
    if folder ==None:
        folder = os.path.join(os.getcwd(), "Train")
    classes = os.listdir(folder)
    signs_per_class = []
    if show_output:
        print(folder)
    for sign in classes:
        sign_folder = os.path.join(folder, sign)
        num_imgs = len(os.listdir(sign_folder))
        signs_per_class.append(num_imgs)
        if show_output:
            print("Class {}: {} images".format(sign, num_imgs))
    return signs_per_class

if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), "nndt_data", "nndt4_unweighted", "white_circular_fc_augmented", "Train")
    folder = os.path.join(os.getcwd(), "Train")
    num_images(folder)