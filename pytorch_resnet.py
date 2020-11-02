import copy
import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torchsummary import summary
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from shutil import copyfile


BATCH_SIZE = 1
INPUT_SIZE = 224
NUM_CLASSES = 43
EPOCHS = 15

# most of this code is adapted from the PyTorch documentation on resnet
# found here: https://pytorch.org/hub/pytorch_vision_resnet/

def make_resnet_model():
    #loads the model architecture

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)

    #modify the last layer to output the number of classes
    model.fc = torch.nn.Linear(512, NUM_CLASSES)
    model.eval()
    return model


def save_model(model, path="pytorch_resnet_saved"):
    #save a model
    path = os.path.join(os.getcwd(), path)
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model(filename):
    # load a model from a file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def generate_training_dataloaders(path = os.getcwd()):
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


def generate_testing_dataloaders(path = os.getcwd()):
    # generates a dataloader for testing a neural net
    # input path should be to root directory of the dataset
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    data_transform = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = datasets.ImageFolder(os.path.join(path, 'Test'), data_transform)
    # Create testing dataloader
    dataloaders_dict = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=4)

    return dataloaders_dict


def get_model_prediction_probs(model, input):
    # feeds an image to a neural network and returns the predictions vector
    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    sm = torch.nn.functional.softmax(output[0], dim=0)
    sm_list = sm.tolist()

    return sm_list


def train_model(model, dataloaders, num_epochs=EPOCHS, lr = 0.001):
    # trains a neural network on the dataloader data
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
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

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
                    print("Completed batch " + str(count))

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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc_history, val_acc_history, training_loss_history, val_loss_history


def plot_accuracy_and_loss(train_acc, train_loss, val_acc, val_loss):
    # plots the training and validation accuracy and loss during training
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(train_acc, label='training accuracy')
    plt.plot(val_acc, label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def create_and_train_model(path = os.getcwd()):
    # creates a model, trains it, plots the accuracy/loss, and saves the model
    model = make_resnet_model()
    # data = generate_reduced_training_dataloader()
    data = generate_training_dataloaders(path = path)
    model, train_acc, val_acc, train_loss, val_loss = train_model(model, data)
    plot_accuracy_and_loss(train_acc, train_loss, val_acc, val_loss)
    save_model(model)
    return model


def test_model_using_dataloader(model, path = os.getcwd(), verbose = False):
    # tests a model using data from a dataloader
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    model.eval()
    test_data_loader = generate_testing_dataloaders(path)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data_loader, 0):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            sample_fname, _ = test_data_loader.dataset.samples[i]
            if verbose:
                print('Model prediction for image {} was {}, label was {}'
                      .format(sample_fname, predicted.item(), labels))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Total number of images tested: {}'.format(str(total)))


def test_one_image(model, image, path = False):
    # gets the neural network prediction vector for a single image
    model.eval()
    if(isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    image = preprocess_image(image, path = path)
    image = create_batch(image.clone().detach())
    preds = get_model_prediction_probs(model, image)
    return preds


def test_model_manually(model, path = None, verbose = False, limit = None, startlimit = None):
    # tests the model on all images in the subfolders of path
    # limit and startlimit allow you to only test the model on a subset of all the images
    # limit specifies how many images to test
    # startlimit specifies the first image to test

    model.eval()
    if (path is not None):
        path = os.path.join(os.getcwd(), path)
    else:
        path = os.getcwd()

    classes = os.listdir(path)

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
            print("Error, invalid limit and startlimit parameters.")
            return -1

    predictions = []

    for i, img in enumerate(imgs):
        #image = Image.open(os.path.join(path, img))
        image = preprocess_image(os.path.join(path, str(labels[i]), img))
        #image = create_batch(torch.tensor(image))
        image = create_batch(image.clone().detach())
        pred = get_model_prediction_probs(model, image)
        class_pred = pred.index(max(pred))
        predictions.append(class_pred)

    if (verbose):
        for i in range(len(imgs)):
            print('Model prediction for image {} was {}, actual was {}'.format(imgs[i], predictions[i], labels[i]))

    # Accuracy with the test data
    total = len(labels)
    correct = 0
    for i in range(total):
        if (labels[i] == predictions[i]):
            correct +=1

    test_accuracy = correct/total
    print('Test Accuracy: {:.5f}%'.format(test_accuracy*100))


def load_and_test_model(path = None, verbose = False):
    # loads a model and tests it using a dataloader and manually

    print("Loading model ...")
    model = load_model(os.path.join(os.getcwd(), "pytorch_resnet_saved"))
    print("Successfully loaded model.")
    test_model_using_dataloader(model, path = path, verbose= verbose)
    test_model_manually(model, path = path, verbose=verbose, limit=10, startlimit=None)


def train_and_test_model_from_scratch(path = os.getcwd(), verbose = False):
    # creates a model, trains it, shows the training data, saves the model, and tests the model
    model = create_and_train_model(path=path)
    test_model_using_dataloader(model, path = path, verbose = verbose)


if __name__ == '__main__':
    load_and_test_model(path = "Debug", verbose=True)