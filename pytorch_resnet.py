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

BATCH_SIZE = 64
INPUT_SIZE = 224
NUM_CLASSES = 43
EPOCHS = 15

# most of this code is adapted from the PyTorch documentation on resnet
# found here: https://pytorch.org/hub/pytorch_vision_resnet/

def make_resnet_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
    model.fc = torch.nn.Linear(512, NUM_CLASSES)
    model.eval()
    return model


def save_model(model, path=os.path.join(os.getcwd(), "pytorch_resnet_saved")):
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model(filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(filename, map_location=device)
    model.eval()
    return model


def get_training_image_files():
    path = os.path.join(os.getcwd(), 'Test')
    images = os.listdir(path)
    filename = os.path.join(path, images[0])
    return filename


def make_val_set(p = 0.2):
    # takes a random subset of the kaggle data and creates a validation set
    # the validation data is removed from the training data
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


def configure_testing_dataset():
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


def preprocess_image(filename):
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor


def create_batch(list_of_tensors):
    input_batch = list_of_tensors.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def generate_reduced_training_dataloader():
    data_transforms = {
        # can apply different transforms to training and validation data, I have chosen not to
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(os.getcwd(), 'small_test_dataset', x),
                                              data_transforms[x]) for x in ['Train', 'Validation']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=BATCH_SIZE, shuffle=True,
                                                       num_workers=4) for x in ['Train', 'Validation']}

    return dataloaders_dict


def generate_training_dataloaders():
    data_transforms = {
        #can apply different transforms to training and validation data, I have chosen not to
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(os.getcwd(), x),
                                              data_transforms[x]) for x in ['Train', 'Validation']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=4) for x in ['Train', 'Validation']}

    return dataloaders_dict


def generate_testing_dataloaders(path = os.getcwd()):
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    data_transform = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = datasets.ImageFolder(os.path.join(path, 'Test'), data_transform)
    # Create training and validation dataloaders
    dataloaders_dict = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=4)

    return dataloaders_dict


def generate_reduced_testing_dataloaders(path = os.path.join(os.getcwd(), "small_test_dataset")):
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    data_transform = transforms.Compose([
        transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(path, 'Test'), data_transform)
    # Create training and validation dataloaders
    dataloaders_dict = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=False, num_workers=4)

    return dataloaders_dict


def get_model_prediction(model, input_batch):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    return output


def get_class_prediction(prediction_output):
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # sm = torch.nn.functional.softmax(prediction_output[0], dim=0)
    # sm_list = sm.tolist()
    _, pred = torch.max(prediction_output, 1)
    # return sm_list.index(max(sm_list))
    return pred


def train_model(model, dataloaders, num_epochs=25, lr = 0.001):
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
                    print("New Batch" + str(count))

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


def create_and_train_model():
    model = make_resnet_model()
    # data = generate_reduced_training_dataloader()
    data = generate_training_dataloaders()
    model, train_acc, val_acc, train_loss, val_loss = train_model(model, data)
    plot_accuracy_and_loss(train_acc, train_loss, val_acc, val_loss)
    save_model(model)
    return model


def test_model(model, path = None):
    '''
    model.eval()
    if path == None:
        path = os.getcwd()
    csv_path = os.path.join(path, "Test.csv")
    y_test = pd.read_csv(csv_path)
    labels = list(y_test["ClassId"].values)
    imgs = y_test["Path"].values

    predictions = []

    for img in imgs:
        #image = Image.open(os.path.join(path, img))
        image = preprocess_image(os.path.join(path, img))
        #image = create_batch(torch.tensor(image))
        image = create_batch(image.clone().detach())
        model_pred = get_model_prediction(model, image)
        class_pred = get_class_prediction(model_pred)
        predictions.append(class_pred)

    #for i in range(len(imgs)):
    #    print('Model prediction for image {} was {}, actual was {}'.format(imgs[i], predictions[i], labels[i]))

    # Accuracy with the test data
    test_accuracy = accuracy_score(labels, predictions)
    print('Test Accuracy: {:.5f}%'.format(test_accuracy*100))
    '''

    #not using this code below for now, it has not been tested
    # code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    model.eval()
    test_data_loader = generate_reduced_testing_dataloaders()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Total number of images tested: {}'.format(str(total)))


def load_and_test_model():
    print("Loading model ...")
    model = load_model(os.path.join(os.getcwd(), "pytorch_resnet_saved"))
    print("Successfully loaded model.")
    test_model(model, os.getcwd())


def train_and_test_model_from_scratch():
    model = create_and_train_model()
    #test_model(model, os.path.join(os.getcwd(), "small_test_dataset"))
    test_model(model, os.getcwd())


if __name__ == '__main__':
    #train_and_test_model_from_scratch()
    #model = load_model("pytorch_resnet_saved")
    #summary(model, (3, 224, 224))
    load_and_test_model()
    #configure_testing_dataset()
    #configure_small_test_dataset_testing_dataset()