import copy
import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torchsummary import summary
from torch.backends import cudnn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from shutil import copyfile
from general import *
from tree_helper import split_all, split_signs, signs
import logging

BATCH_SIZE = 64
INPUT_SIZE = 224
NUM_CLASSES = 43
EPOCHS = 10
GPU_ID = 0

def set_gpu_id(gpu_id):
    GPU_ID = gpu_id

def set_cuda(gpu_id = None):
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        cudnn.benchmark = True

# most of this code is adapted from the PyTorch documentation on resnet
# found here: https://pytorch.org/hub/pytorch_vision_resnet/

def make_resnet_model(num_classes = NUM_CLASSES):
    #loads the model architecture

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)

    #modify the last layer to output the number of classes
    model.fc = torch.nn.Linear(512, num_classes)
    model.eval()
    return model


def save_model(model, path=None):
    #save a model
    if path == None:
        date = str_date()
        filename = "resnet" + date
        path = os.path.join(os.getcwd(), "Models", filename)
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model(filename, gpu_id = GPU_ID):
    # load a model from a file
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model = torch.load(filename, map_location=device)
    model.eval()
    return model


def make_val_set(p = 0.2):
    # takes a random subset of the kaggle data and creates a validation set
    # the validation images are removed from the training data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.getcwd()
    classes = os.listdir(os.path.join(PATH, "Train"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Validation", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Train", sign))
        random.shuffle(list_of_files)
        n_idxs = int(len(list_of_files)*p)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            os.rename(os.path.join(PATH, "Train", sign, file), os.path.join(PATH, "Validation", sign, file))


def make_debug_set(proportion_or_number = 0.2):
    # takes a random subset of the kaggle data and creates a debugging set for testing
    # the debug images are copied from the Testing data, this function does not remove
    # anything from the testing data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.getcwd()
    classes = os.listdir(os.path.join(PATH, "Test"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Debug", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Test", sign))
        random.shuffle(list_of_files)
        if proportion_or_number > 1:
            n_idxs = proportion_or_number
        else:
            n_idxs = int(len(list_of_files)*proportion_or_number)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            copyfile(os.path.join(PATH, "Test", sign, file), os.path.join(PATH, "Debug", sign, file))


def format_folder_two_digits(path):
    # formats all subdirectories in 'path' to be 2 digits long
    # example: a folder that was previously named '9' would be renamed to '09'
    for folder_name in os.listdir(path):
        if len(folder_name) != 2:
            os.rename(os.path.join(path, folder_name), os.path.join(path, '0' + folder_name))


def configure_testing_dataset():
    # organizes a list of random test images into folders where each folder is a class of images
    # example: instead of having all test images under /Test/, organize them into subfolders called
    # /Test/x/ where x ranges from 0 to 42 and represents the 43 different road signs
    PATH = os.getcwd()
    classes = os.listdir(os.path.join(PATH, "Train"))

    csv_path = os.path.join(PATH, "Test.csv")
    y_test = pd.read_csv(csv_path)
    labels = list(y_test["ClassId"].values)
    imgs = y_test["Path"].values

    for sign in classes:
        os.makedirs(os.path.join(PATH, "Test", sign), exist_ok=True)

    for i, img in enumerate(imgs):
        #img is the filename of the image
        label = labels[i]
        newimage = img.split("/")[1]
        os.rename(os.path.join(PATH, img), os.path.join(PATH, 'Test', str(label), newimage))


def configure_small_test_dataset_testing_dataset():
    # same function as configure_testing_dataset() except for the small test dataset
    PATH = os.path.join(os.getcwd(), "small_test_dataset")
    classes = os.listdir(os.path.join(PATH, "Train"))

    csv_path = os.path.join(PATH, "Test.csv")
    y_test = pd.read_csv(csv_path)
    labels = list(y_test["ClassId"].values)
    imgs = y_test["Path"].values

    for sign in classes:
        os.makedirs(os.path.join(PATH, "Test", sign), exist_ok=True)

    for i, img in enumerate(imgs):
        # img is the filename of the image
        label = labels[i]
        newimage = img.split("/")[1]
        os.rename(os.path.join(PATH, img), os.path.join(PATH, 'Test', str(label), newimage))


def preprocess_image(image, path = True):
    # processes an image which is necessary before feeding the image to the neural network
    if(path):
        image = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    return input_tensor


def create_batch(list_of_tensors):
    input_batch = list_of_tensors.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def generate_training_dataloaders(path = None):
    if (path == None):
        path = os.getcwd()
    # creates and returns a dataloader used for neural network training
    data_transforms = {
        # can apply different transforms to training and validation data, this could be used to augment the
        # training data; I have chosen not to
        'Train': transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                              data_transforms[x]) for x in ['Train', 'Validation']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=4) for x in ['Train', 'Validation']}

    return dataloaders_dict


def generate_testing_dataloaders(test_path = None):
    if (test_path == None):
        test_path = os.path.join(os.getcwd(), "Test")
    # generates a dataloader for testing a neural net
    # input path should be to root directory of the dataset
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    data_transform = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = datasets.ImageFolder(test_path, data_transform)
    # Create testing dataloader
    dataloaders_dict = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=4)

    return dataloaders_dict


def get_model_prediction_probs(model, input, gpu_id = None):
    # feeds an image to a neural network and returns the predictions vector
    # gpu_id: integer of the gpu to use (if available)
    if torch.cuda.is_available():
        if gpu_id is not None:
            model.cuda(gpu_id)
            input = input.to(gpu_id)
        else:
            input = input.to('cuda')
            model.to('cuda')

    with torch.no_grad():
        output = model(input)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    sm = torch.nn.functional.softmax(output[0], dim=0)
    sm_list = sm.tolist()

    return sm_list


def train_model(model, dataloaders, folder, num_epochs=EPOCHS, lr = 0.001, save = True):
    # trains a neural network on the dataloader data
    # model(pytorch object): neural network to train
    # dataloaders (dict {"Train": <dataloader of training data>, "Validation": <dataloader of validation data>})
    # folder: folder to save training output log to

    # setup logging
    logger = logging.getLogger("neural_net_training")
    logger.setLevel(logging.INFO)
    logfile = os.path.join(folder, "training_output.txt")
    if save:
        logging.basicConfig(filename=logfile, format='%(message)s')
    else:
        logging.basicConfig(format='%(message)s')
    logging.getLogger("neural_net_training").addHandler(logging.StreamHandler())
    logger.info("Training from folder {}".format(folder))

    # this code is adapted from the PyTorch tutorial
    # at https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    since = time.time()

    val_loss_history = []
    val_acc_history = []
    training_loss_history = []
    training_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                count += 1
                if (count % 100 == 0):
                    logger.info("Completed batch " + str(count))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                training_acc_history.append(epoch_acc)
                training_loss_history.append(epoch_loss)
            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'Validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best Validation Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc_history, val_acc_history, training_loss_history, val_loss_history


def plot_accuracy_and_loss(train_acc, train_loss, val_acc, val_loss, path = None):
    # plots the training and validation accuracy and loss during training
    # plotting graphs for accuracy
    # if path is specified, saves the plots under path folder
    plt.figure(0)
    plt.plot(train_acc, label='training accuracy')
    plt.plot(val_acc, label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    if path is not None:
        fname = os.path.join(path, "accuracy.png")
        plt.savefig(fname)
    else:
        plt.show()

    plt.figure(1)
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if path is not None:
        fname = os.path.join(path, "loss.png")
        plt.savefig(fname)
    else:
        plt.show()


def create_and_train_model(path = None, filename = None, num_classes = NUM_CLASSES):
    if (path == None):
        path = os.getcwd()
    # creates a model, trains it, plots the accuracy/loss, and saves the model
    model = make_resnet_model(num_classes)
    # data = generate_reduced_training_dataloader()
    data = generate_training_dataloaders(path = path)
    model, train_acc, val_acc, train_loss, val_loss = train_model(model, data, path)
    plot_accuracy_and_loss(train_acc, train_loss, val_acc, val_loss, path = path)
    if filename is not None:
        save_model(model, filename)
    else:
        save_model(model)
    return model


def test_model_using_dataloader(model, path = None, verbose = False):
    if (path == None):
        path = os.path.join(os.getcwd(), "Test")
    # tests a model using data from a dataloader
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    model.eval()
    test_data_loader = generate_testing_dataloaders(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions += list(predicted)
    if verbose:
        for i in range(len(test_data_loader.dataset.samples)):
            img_path, true_class = test_data_loader.dataset.samples[i]
            img_file = os.path.split(img_path)[1]
            print('Model prediction for image {} was {}, actual was {}'
                  .format(img_file, predictions[i], true_class))
    print('Test Accuracy: {:.7f}%'.format(100 * correct / total))
    print('Total number of images tested: {}'.format(str(total)))


def test_one_image(model, image, path = False, gpu_id = None):
    # gets the neural network prediction vector for a single image
    # gpu_id: integer of the gpu to use (if available)
    model.eval()
    if(isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    image = preprocess_image(image, path = path)
    image = create_batch(image.clone().detach())
    preds = get_model_prediction_probs(model, image, gpu_id = gpu_id)
    return preds


def test_model_manually(model, path = None, verbose = False, limit = None, startlimit = None, nndt = False,
                        exclusive = None, top_misclassifications = None, byclass = False, save_file = None):
    # tests the model on all images in the subfolders of path
    # limit and startlimit allow you to only test the model on a subset of all the images
    # limit specifies how many images to test
    # startlimit specifies the first image to test
    # nndt is a boolean indicating if the input model is an nndt object or a neural network
    # if exclusive is not none it should be an integer array representing the classes that should be tested on the
    # model.  For example, exclusive = [0, 1] will only test the model on signs of class 0 and 1
    # byclass indicates if results should be printed out by class
    # top_misclassifications should be an int if none and prints out the most frequent misclassifications for each
    # class.  This can only be used if byclass is True

    if not nndt:
        model.eval()
    if (path == None):
        path = os.path.join(os.getcwd(), "Test")

    output_str = "Testing Parameters: \n"
    output_str += "----------------------------------------\n"
    output_str += "NNDT:            {}\n".format(str(nndt))
    output_str += "Test_folder:     {}\n".format(path)
    output_str += "Limit:           {}\n".format(str(limit))
    output_str += "Exclusive:       {}\n".format(str(exclusive))
    output_str += "By Class:        {}\n".format(str(byclass))
    output_str += "Top Misclass:    {}\n".format(str(top_misclassifications))
    output_str += "Save File:       {}\n\n".format(str(save_file))

    save = save_file is not None
    if save:
        print(save_file)
        f = open(save_file, "w+")
        f.write(output_str)
    else:
        print(output_str)

    classes = os.listdir(path)
    if exclusive is not None:
        classes = [format_two_digits(x) for x in exclusive]

    if byclass == True:
        # imgs_labels is a dict where the keys are the classes and the values are arrays of paths to images of that class
        imgs_labels = {}

        # img_results is an array of 3-tuples of the format
        # (class_name, number of images tested of that class, number of correctly classified images)
        img_results = []

        # initialize misclassifications if specified in function call
        if top_misclassifications is not None:
            # dict of dicts for misclassifications, format is
            # {class name: {misclassified class: number of misclassifications}}
            misclassified = {}
            for sign in classes:
                misclassified[int(sign)] = {}

        # get paths to images
        for sign in classes:
            sign_directory = os.path.join(path, str(sign))
            # array of full path to image
            test_data = [os.path.join(path, sign, x) for x in os.listdir(sign_directory)]
            imgs_labels[int(sign)] = test_data

        # reduce number of images if specified in the function call
        if limit is not None:
            for key in imgs_labels:
                imgs_labels[key] = imgs_labels[key][:limit]

        # test the images
        total_images = 0
        total_correct = 0
        for key in imgs_labels:  # key is the integer version of the original class
            class_images_count = len(imgs_labels[key])
            class_correct_count = 0
            total_images += class_images_count

            if verbose:
                output_str = "Class {}\n".format(key)
                if save:
                    f.write(output_str)
                else:
                    print(output_str)

            for i, img in enumerate(imgs_labels[key]):
                if nndt:
                    pred_vector = model.prediction_vector(img)
                    pred = model.prediction(pred_vector)
                    class_pred = int(pred[0])
                else:
                    image = preprocess_image(img)
                    image = create_batch(image.clone().detach())
                    pred = get_model_prediction_probs(model, image)
                    class_pred = pred.index(max(pred))
                if class_pred == int(key):
                    class_correct_count += 1
                elif top_misclassifications is not None:
                    if class_pred in misclassified[key]:
                        misclassified[key][class_pred] += 1
                    else:
                        misclassified[key][class_pred] = 1
                if (verbose):
                    output_str = '({}) Model prediction for image {} was {}, actual was {}\n'.format(str(i + 1),
                                                                                            "..." + str(img[-30:]),
                                                                                            str(class_pred),
                                                                                            str(key))
                    if save:
                        f.write(output_str)
                    else:
                        print(output_str)

            if verbose:
                output_str = "\n\n"
                if save:
                    f.write(output_str)
                else:
                    print(output_str)

            img_results.append((key, class_images_count, class_correct_count))
            total_correct += class_correct_count

        for i in range(len(img_results)):
            sign_class, count, correct = img_results[i]
            output_str = '\nClass {}: {}/{} images correct, {:4f}% accuracy\n'.format(sign_class, correct, count,
                                                                             correct * 100 / count)
            if save:
                f.write(output_str)
            else:
                print(output_str)
            if top_misclassifications is not None and count != correct:
                sign_misclassifications = misclassified[sign_class]
                sorted_misclassifications = sorted(sign_misclassifications.items(), key=lambda x: x[1], reverse=True)
                sorted_misclassifications = sorted_misclassifications[:top_misclassifications]
                output_str = '    Top {} misclassifications:\n'.format(top_misclassifications)
                if save:
                    f.write(output_str)
                else:
                    print(output_str)
                for misclassified_class, freq in sorted_misclassifications:
                    output_str = '        Class {}, {} occurances, ({:4f}% of all misclassifications)'.format(
                        misclassified_class, freq, 100 * freq / (count - correct))
                    if save:
                        f.write(output_str)
                    else:
                        print(output_str)

        test_accuracy = total_correct / total_images
        output_str = '\n\nOverall test Accuracy: {:.7f}%\n'.format(test_accuracy * 100)
        output_str += "{}/{} images correct".format(total_correct, total_images)
        if save:
            f.write(output_str)
            f.close()
        else:
            print(output_str)
        return
    #----------------------------------------------------------------------------------
    imgs = []
    labels = []

    for sign in classes:
        sign_directory = os.path.join(path, str(sign))
        test_data = os.listdir(sign_directory)
        labels.extend([sign]*len(test_data))
        imgs.extend(test_data)

    if limit is not None:
        start = 0
        if startlimit is not None:
            start = startlimit
        labels = labels[start:limit+start]
        imgs = imgs[start:limit+start]
        if (len(labels)<1 or len(imgs)<1):
            output_str = "Error, invalid limit and startlimit parameters."
            if save:
                f.write(output_str)
            else:
                print(output_str)
            return -1

    predictions = []

    for i, img in enumerate(imgs):
        full_file = os.path.join(path, str(labels[i]), img)
        if nndt:
            pred_vector = model.prediction_vector(full_file)
            pred = model.prediction(pred_vector)
            predictions.append(pred[0])
        else:
            #image = Image.open(os.path.join(path, img))
            image = preprocess_image(full_file)
            #image = create_batch(torch.tensor(image))
            image = create_batch(image.clone().detach())
            pred = get_model_prediction_probs(model, image)
            class_pred = pred.index(max(pred))
            predictions.append(class_pred)

    if (verbose):
        for i in range(len(imgs)):
            output_str = 'Model prediction for image {} was {}, actual was {}\n'.format(str(imgs[i]),
                                                                               predictions[i], labels[i])
            if save:
                f.write(output_str)
            else:
                print(output_str)

    # Accuracy with the test data
    total = len(labels)
    correct = 0
    for i in range(total):
        if (int(labels[i]) == int(predictions[i])):
            correct +=1

    test_accuracy = correct/total
    output_str = 'Test Accuracy: {:.7f}%\n'.format(test_accuracy*100)
    output_str += "Number of images tested: {}\n".format(str(total))
    if save:
        f.write(output_str)
        f.close()
    else:
        print(output_str)


def test_attribute_model_manually(model, attribute, correct_classes, mapping, path = None, verbose = False, limit = None,
                                  exclusive = None, byclass = False, save_file = None):
    # tests an attribute model on all images in the subfolders of path, where the sufolders are organized by sign class
    # like in the original Training data folder
    # limit specifies how many images of each class to test
    # exclusive is an array of ints specifying which images to test
    # mapping is an array of the length of the output of the model, and specifies what attribute each index refers to;
    # int of model prediction -> string of attribute
    # correct_classes is an array of ints specifying which classes should reach this node and distinguishes the classes
    # in the none node
    # save_file specifies a path to save the output in

    output_str = "Testing Parameters: \n"
    output_str += "----------------------------------------\n"
    output_str += "Attribute:       {}\n".format(attribute)
    output_str += "Correct classes: {}\n".format(str(correct_classes))
    output_str += "Mapping:         {}\n".format(str(mapping))
    output_str += "Test_folder:     {}\n".format(path)
    output_str += "Limit:           {}\n".format(str(limit))
    output_str += "Exclusive:       {}\n".format(str(exclusive))
    output_str += "By Class:        {}\n".format(str(byclass))
    output_str += "Save File:        {}\n".format(str(save_file))


    save = save_file is not None
    if save:
        print(save_file)
        f = open(save_file, "w+")
        f.write(output_str)
    else:
        print(output_str)

    model.eval()
    if (path == None):
        path = os.path.join(os.getcwd(), "Test")

    classes = os.listdir(path)
    if exclusive is not None:
        classes = [format_two_digits(x) for x in exclusive]

    # array where each index represents a class, and the value at that index represents the attribute label for that
    # class
    class_attributes = split_all(attribute)

    # class_labels is an array of where the index represents the class (0 to 42) and value at that index represents the
    # attribute of that class (either "none" or attribute string)
    class_labels = [class_attributes[i] if i in correct_classes else "none" for i in range(43)]


    # imgs_labels is a dict where the keys are the classes and the values are arrays of paths to images of that class
    imgs_labels = {}

    # img_results is an array of 3-tuples of the format
    # (class_name, number of images tested of that class, number of correctly classified images)
    img_results = []

    for sign in classes:
        sign_directory = os.path.join(path, str(sign))
        # array of full path to image
        test_data = [os.path.join(path, sign, x) for x in os.listdir(sign_directory)]
        if limit is not None:
            test_data = test_data[:limit]
        imgs_labels[sign] = test_data

    # test the images
    total_images = 0
    total_correct = 0
    for key in imgs_labels:  # key is the integer version of the original class
        class_images_count = len(imgs_labels[key])
        class_correct_count = 0
        total_images += class_images_count

        if verbose:
            output_str = "\nClass {}\n".format(key)
            if save:
                f.write(output_str)
            else:
                print(output_str)

        for i, img in enumerate(imgs_labels[key]):
            image = preprocess_image(img)
            image = create_batch(image.clone().detach())
            pred = get_model_prediction_probs(model, image)
            numerical_pred = pred.index(max(pred))
            attribute_pred = mapping[numerical_pred]
            if attribute_pred == "False":
                attribute_pred = False
            if attribute_pred == "True":
                attribute_pred = True
            if attribute_pred == class_labels[int(key)]:
                class_correct_count += 1
            if (verbose):
                output_str = '({}) Model prediction for image {} was {}, actual was {} (true class {})\n'.format(str(i+1),
                                                                                                        "..." + str(img[-30:]),
                                                                                                        str(attribute_pred),
                                                                                                        str(class_labels[int(key)]),
                                                                                                        str(key))
                if save:
                    f.write(output_str)
                else:
                    print(output_str)


        if verbose:
            output_str = "\n\n"
            if save:
                f.write(output_str)
            else:
                print(output_str)

        img_results.append((key, class_images_count, class_correct_count))
        total_correct += class_correct_count

    if byclass:
        for i in range(len(img_results)):
            sign_class, count, correct = img_results[i]
            output_str = '\nClass {}: {}/{} images correct, {:4f}% accuracy'.format(sign_class, correct, count,
                                                                             correct * 100 / count)
            if save:
                f.write(output_str)
            else:
                print(output_str)

    test_accuracy = total_correct / total_images
    output_str = '\n\nOverall test Accuracy: {:.7f}%\n'.format(test_accuracy * 100)
    output_str += "{}/{} images correct\n\n".format(total_correct, total_images)
    if save:
        f.write(output_str)
        f.close()
    else:
        print(output_str)

    return


def test_final_classifier_manually(model, road_signs, path = None, verbose = False, limit = None,
                                   startlimit = None, exclusive = None, save_file = None):
    # tests a final classifier on all images in the subfolders of path, where the sufolders are organized by
    # sign class like in the original Training data folder.
    # <road_signs> are the signs included in the final classifier, all other road signs will be grouped into the
    # "none" category
    # limit and startlimit allow you to only test the model on a subset of all the images
    # limit specifies how many images to test
    # startlimit specifies the first image to test
    # save_file specifies a path to save the output in

    output_str = "Testing Parameters: \n"
    output_str += "----------------------------------------\n"
    output_str += "Classes in FC:   {}\n".format(str(road_signs))
    output_str += "Test_folder:     {}\n".format(path)
    output_str += "Limit:           {}\n".format(str(limit))
    output_str += "Exclusive:       {}\n".format(str(exclusive))
    output_str += "By Class:        False\n"
    output_str += "Save File:        {}\n".format(str(save_file))

    save = save_file is not None
    if save:
        print(save_file)
        f = open(save_file, "w+")
        f.write(output_str)
    else:
        print(output_str)

    model.eval()
    if (path == None):
        path = os.path.join(os.getcwd(), "Test")

    classes = os.listdir(path)
    if exclusive is not None:
        classes = [format_two_digits(x) for x in exclusive]

    # map class label to attribute label for predicted output
    mapping = {}
    for i, sign in enumerate(road_signs):
        mapping[sign] = i
    mapping["none"] = len(road_signs)


    # array where each index represents a class, and the value at that index represents the attribute label for that
    # class
    class_attributes = []
    for i in range(43):
        if i in road_signs:
            class_attributes.append(i)
        else:
            class_attributes.append("none")

    imgs = []
    labels = []

    for sign in classes:
        sign_directory = os.path.join(path, str(sign))
        # array of full path to image
        test_data = [os.path.join(path, sign, x) for x in os.listdir(sign_directory)]
        # labels are the string versions of the attribute values
        labels.extend([class_attributes[int(sign)]]*len(test_data))
        imgs.extend(test_data)

    if limit is not None:
        start = 0
        if startlimit is not None:
            start = startlimit
        labels = labels[start:limit+start]
        imgs = imgs[start:limit+start]
        if (len(labels)<1 or len(imgs)<1):
            print("Error, invalid limit and startlimit parameters.")
            return -1

    predictions = []

    for i, img in enumerate(imgs):
        #image = Image.open(os.path.join(path, img))
        image = preprocess_image(img)
        #image = create_batch(torch.tensor(image))
        image = create_batch(image.clone().detach())
        pred = get_model_prediction_probs(model, image)
        class_pred = pred.index(max(pred))
        predictions.append(class_pred)

    if (verbose):
        for i in range(len(imgs)):
            output_str = '({}) Model prediction for image {} was {}, actual was {} ({})\n'.format(str(i),
                                                                                                  "..." + str(imgs[i][-30:]),
                                                                                                  predictions[i],
                                                                                                  labels[i],
                                                                                                  str(mapping[labels[i]]))
            if save:
                f.write(output_str)
            else:
                print(output_str)

    # Accuracy with the test data
    total = len(labels)
    correct = 0
    for i in range(total):
        if (mapping[labels[i]] == int(predictions[i])):
            correct +=1

    test_accuracy = correct/total
    output_str = '\n\nOverall test Accuracy: {:.7f}%\n'.format(test_accuracy * 100)
    output_str += "{}/{} images correct\n\n".format(correct, total)
    if save:
        f.write(output_str)
        f.close()
    else:
        print(output_str)


def test_final_classifier_manually_byclass(model, road_signs, path = None, verbose = False, limit = None,
                                           exclusive = None, top_misclassifications = None, save_file = None):
    # tests a final classifier on all images in the subfolders of path, where the sufolders are organized by
    # sign class like in the original Training data folder.  Returns accuracy on each class
    # <road_signs> are the signs included in the final classifier, all other road signs will be grouped into the
    # "none" category
    # limit specifies how many images to test for each class
    # top_misclassifications is an int; if specified, prints the top classes that each class was misclassified as
    # save_file specifies a path to save the output in

    output_str = "Testing Parameters: \n"
    output_str += "----------------------------------------\n"
    output_str += "Classes in FC:   {}\n".format(str(road_signs))
    output_str += "Test_folder:     {}\n".format(path)
    output_str += "Limit:           {}\n".format(str(limit))
    output_str += "Exclusive:       {}\n".format(str(exclusive))
    output_str += "By Class:        True\n"
    output_str += "Top Misclass:    {}\n".format(str(top_misclassifications))
    output_str += "Save File:       {}\n".format(str(save_file))

    save = save_file is not None
    if save:
        print(save_file)
        f = open(save_file, "w+")
        f.write(output_str)
    else:
        print(output_str)


    # set up model and get test folder
    model.eval()
    if (path == None):
        path = os.path.join(os.getcwd(), "Test")

    # get list of classes to test
    classes = os.listdir(path)
    if exclusive is not None:
        classes = [format_two_digits(x) for x in exclusive]

    # map class label to attribute label for predicted output
    mapping = {}
    inv_mapping = {}
    for i, sign in enumerate(road_signs):
        mapping[sign] = i
        inv_mapping[i] = sign
    mapping["none"] = len(road_signs)
    inv_mapping[len(road_signs)] = "none"


    # imgs_labels is a dict where the keys are the classes and the values are arrays of paths to images of that class
    imgs_labels = {}

    # img_results is an array of 3-tuples of the format
    # (class_name, number of images tested of that class, number of correctly classified images)
    img_results = []

    # initialize misclassifications if specified in function call
    if top_misclassifications is not None:
        # dict of dicts for misclassifications, format is
        # {class name: {misclassified class: number of misclassifications}}
        misclassified = {}
        for sign in classes:
            misclassified[int(sign)] = {}

    # get paths to images
    for sign in classes:
        sign_directory = os.path.join(path, str(sign))
        # array of full path to image
        test_data = [os.path.join(path, sign, x) for x in os.listdir(sign_directory)]
        imgs_labels[int(sign)] = test_data

    # reduce number of images if specified in the function call
    if limit is not None:
        for key in imgs_labels:
            imgs_labels[key] = imgs_labels[key][:limit]

    # test the images
    total_images = 0
    total_correct = 0
    for key in imgs_labels: # key is the integer version of the original class
        class_images_count = len(imgs_labels[key])
        class_correct_count = 0
        total_images += class_images_count

        if verbose:
            output_str = "\nClass {}\n".format(key)
            if save:
                f.write(output_str)
            else:
                print(output_str)

        for i, img in enumerate(imgs_labels[key]):
            image = preprocess_image(img)
            image = create_batch(image.clone().detach())
            pred = get_model_prediction_probs(model, image)
            class_pred = pred.index(max(pred))
            fc_label = mapping[key] if key in mapping else mapping['none']
            if class_pred == fc_label:
                class_correct_count += 1
            elif top_misclassifications is not None:
                if inv_mapping[class_pred] in misclassified[key]:
                    misclassified[key][inv_mapping[class_pred]] +=1
                else:
                    misclassified[key][inv_mapping[class_pred]] = 1
            if (verbose):
                output_str = '({}) Model prediction for image {} was {} ({}), actual was {} (true class {})\n'.format(
                                                                                             str(i+1),
                                                                                             "..." + str(img[-30:]),
                                                                                             str(class_pred),
                                                                                            "class " + str(inv_mapping[class_pred]),
                                                                                             str(fc_label),
                                                                                             str(key))
                if save:
                    f.write(output_str)
                else:
                    print(output_str)

        if verbose:
            output_str = "\n\n"
            if save:
                f.write(output_str)
            else:
                print(output_str)

        img_results.append((key, class_images_count, class_correct_count))
        total_correct += class_correct_count

    for i in range(len(img_results)):
        sign_class, count, correct = img_results[i]
        output_str = '\nClass {}: {}/{} images correct, {:4f}% accuracy\n'.format(sign_class, correct, count, correct * 100 / count)
        if save:
            f.write(output_str)
        else:
            print(output_str)

        if top_misclassifications is not None and count != correct:
            sign_misclassifications = misclassified[sign_class]
            sorted_misclassifications = sorted(sign_misclassifications.items(), key=lambda x: x[1], reverse=True)
            sorted_misclassifications = sorted_misclassifications[:top_misclassifications]
            output_str = '    Top {} misclassifications:\n'.format(top_misclassifications)
            if save:
                f.write(output_str)
            else:
                print(output_str)
            for misclassified_class, freq in sorted_misclassifications:
                output_str = '        Class {}, {} occurances, ({:4f}% of all misclassifications)\n'.format(misclassified_class,
                                                                                                freq,
                                                                                                100*freq/(count-correct))
                if save:
                    f.write(output_str)
                else:
                    print(output_str)


    test_accuracy = total_correct/total_images
    output_str = '\n\nOverall test Accuracy: {:.7f}%\n'.format(test_accuracy * 100)
    output_str += "{}/{} images correct\n\n".format(total_correct, total_images)
    if save:
        f.write(output_str)
        f.close()
    else:
        print(output_str)


def load_and_test_model(modelpath, test_path = None, verbose = False):
    # loads a model and tests it using a dataloader and manually

    print("Loading model " + modelpath + "...")
    model = load_model(os.path.join(os.getcwd(), modelpath))
    print("Successfully loaded model.")
    test_model_using_dataloader(model, path = test_path, verbose= verbose)
    test_model_manually(model, path = test_path, verbose=verbose, startlimit=None)


def train_and_test_model_from_scratch(path = None, verbose = False):
    if(path == None):
        path = os.getcwd()
    # creates a model, trains it, shows the training data, saves the model, and tests the model
    model = create_and_train_model(path=path)
    test_model_using_dataloader(model, path = path, verbose = verbose)


if __name__ == '__main__':
    #load_and_test_model("pytorch_resnet_saved_11_9_20", test_path=os.path.join(os.getcwd(), 'Debug'), verbose=True)
    # train_and_test_model_from_scratch()
    # model_path = os.path.join(os.getcwd(), "nndt_data", "nndt3_unweighted", "triangle_final_classifier",
    #                           "triangle_final_classifier_resnet_2021-01-05")
    model_path = os.path.join(os.getcwd(), "Models", "pytorch_resnet_saved_11_9_20")
    model = load_model(model_path)
    # print(summary(model, input_size=(3, INPUT_SIZE, INPUT_SIZE)))
    path = os.path.join(os.getcwd(), "small_test_dataset")
    #create_and_train_model(path = path, filename="TEST_RESNET_DELETEME", num_classes=2)
    image = os.path.join(os.getcwd(), "test_images", "pxArt.png")
    a = test_one_image(model, image, path=True)
    print(a)