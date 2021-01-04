import torch.nn.functional as F
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from image_features import gather_image_features
import numpy as np


# input size is the length of the feature vector, see gather_image_features in image_features.py
INPUT_SIZE = 22
HIDDEN_LAYER_SIZE = 100
NUM_CLASSES = 43
EPOCHS = 200
LEARNING_RATE = 0.1


class Net(nn.Module):
    # adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        self.fc4 = nn.Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES)
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #return torch.sigmoid(self.fc4(x))
        #return self.fc4(x)
        return F.relu(self.fc4(x))


def train(csv_file, epochs = EPOCHS, lr = LEARNING_RATE):
    #adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    df = pd.read_csv(csv_file)
    X = df[['max_color_len', 'max_color_r', 'max_color_g', 'max_color_b', 'r_avg', 'g_avg', 'b_avg',
            'dom_color1_r', 'dom_color1_g', 'dom_color1_b', 'dom_color2_r', 'dom_color2_g', 'dom_color2_b',
            'dom_color3_r', 'dom_color3_g', 'dom_color3_b', 'dom_color4_r', 'dom_color4_g', 'dom_color4_b',
            'dom_color5_r', 'dom_color5_g', 'dom_color5_b']]
    y = df[['img_class']]

    # split data into 20% validation and 80% training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # 2d tensor
    X_train = torch.from_numpy(X_train.to_numpy()).float()
    # 1d tensor
    y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()))
    # 2d tensor
    X_val = torch.from_numpy(X_val.to_numpy()).float()
    # 1d tensor
    y_val = torch.squeeze(torch.from_numpy(y_val.to_numpy()))

    since = time.time()
    val_loss_history = []
    val_acc_history = []
    training_loss_history = []
    training_acc_history = []

    model = Net()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr)#, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        # divide learning rate in half every 25 epochs
        if(epochs % 100 == 0 and epoch != 0):
            lr = lr/2

        #----------training data----------
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # run training data through the model and get predictions
            y_pred = model(X_train)
            y_pred = torch.squeeze(y_pred)

            # calculate training loss and add it to the history, do the same with training accuracy
            train_loss = criterion(y_pred, y_train)
            training_loss_history.append(train_loss)
            train_acc = calculate_accuracy(y_train, y_pred)
            training_acc_history.append(train_acc)

            # update model based on training data
            train_loss.backward()
            optimizer.step()

        #----------validation data----------
        model.eval()
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            # run validation data through the model and get predictions
            y_val_pred = model(X_val)
            y_val_pred = torch.squeeze(y_val_pred)

            # calculate validation loss and add it to the history, do the same with validation accuracy
            val_loss = criterion(y_val_pred, y_val)
            val_loss_history.append(val_loss)
            val_acc = calculate_accuracy(y_val, y_val_pred)
            val_acc_history.append(val_acc)

        print("epoch {}\nTrain set - loss: {:.4f}, accuracy: {:.4f}%\nVal   set - loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(str(epoch), train_loss.item(), train_acc.item()*100, val_loss.item(), val_acc.item()*100))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, training_acc_history, val_acc_history, training_loss_history, val_loss_history


def calculate_accuracy(y_true, y_pred):
    # y_true: 1d tensor of sample predictions
    # y_pred: 2d tensor where the 1st dimension is the sample and the second dimension is the prediction
    # vector for that sample
    a = torch.max(y_pred, dim=1)[1]
    print(a[:10])
    b = y_true
    print(b[:10])
    c = a==b
    print(c[:10])
    return c.sum().float()/len(y_true)


def get_model_prediction(model, data):
    # adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(data).float().to(device)
    output = model(t)
    return output.ge(0.5).item(), output


def save_model(model, path="pytorch_saved"):
    #save a model
    path = os.path.join(os.getcwd(), "ML", path)
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model():
    # load a model from a file
    filename = os.path.join(os.getcwd(), "ML", "pytorch_saved")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(filename)
    model.to(device)
    model.eval()
    return model


def create_train_save_print_model(csv):
    model, training_acc, test_acc, training_loss, test_loss = train(csv)
    model.eval()
    image = os.path.join(os.getcwd(), 'Test', '00', '00243.png')
    features = gather_image_features(image)
    features = torch.from_numpy(np.array(features)).float()
    output = model(features)
    print(output)
    #save_model(model)
    #plot_accuracy_and_loss(training_acc, training_loss, test_acc, test_loss)


def plot_accuracy_and_loss(train_acc, train_loss, test_acc, test_loss):
    # plots the training and validation accuracy and loss during training
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(train_acc, label='training accuracy')
    plt.plot(test_acc, label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    path = os.path.join(os.getcwd(), "ML", "Outputs", "accuracy.png")
    plt.savefig(path)

    plt.figure(1)
    plt.plot(train_loss, label='training loss')
    plt.plot(test_loss, label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    path = os.path.join(os.getcwd(), "ML", "Outputs", "loss.png")
    plt.savefig(path)

csv = os.path.join(os.getcwd(), "Image_features", "img_features_GTSRB_ResNet_2020-12-29_normalized.csv")
#csv = os.path.join(os.getcwd(), "Image_features", "img_features_small_test_dataset_2020-12-29.csv")
#train(csv)
create_train_save_print_model(csv)