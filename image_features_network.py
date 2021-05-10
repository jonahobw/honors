import copy

import torch.nn.functional as F
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

from general import str_date
from image_features import gather_image_features, load_hog_df
import numpy as np


# input size is the length of the feature vector, see gather_image_features in image_features.py
from tree_helper import split_all

INPUT_SIZE = 22
HIDDEN_LAYER_SIZE = 200
NUM_CLASSES = 43
EPOCHS = 5
LEARNING_RATE = 0.01


class hog_dataset(Dataset):
    # adapted from https://medium.com/@shashikachamod4u/excel-csv-to-pytorch-dataset-def496b6bcc1

    def __init__(self, filename):
        global NUM_CLASSES
        # read csv and load data
        # specific to hog -----------------------------------
        df = pd.read_csv(filename)
        x = df.iloc[0:len(df.index), 2:82].values
        y = df.iloc[0:len(df.index), 1].values
        # x = df.iloc[0:256, 2:82].values
        # y = df.iloc[0:256, 1].values
        # array where each index represents a class, and the value at that index represents the attribute label for that
        # class
        class_attributes = split_all("shape")
        y = [class_attributes[i] for i in y]
        mapping = {"circle": 0, "triangle": 1, "diamond": 2, "inverted_triangle" : 3, "octagon" : 4}
        y = [mapping[i] for i in y]
        num_unique_classes = len(list(set(y)))
        NUM_CLASSES = num_unique_classes

        # Feature Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # converting  to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return int(len(self.y_train))

    def __getitem__(self, item):
        return self.X_train[item], self.y_train[item]


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


def save_model(model, fname = None):
    #save a model
    if fname == None:
        path = os.path.join(os.getcwd(), "Image_features", fname)
    torch.save(model, fname)
    print('Model saved as {}'.format(fname))
    return fname


def load_model(filename = None):
    # load a model from a file
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


def train_hog_model(model, dataloader, lr = LEARNING_RATE, epochs = EPOCHS):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    since = time.time()

    num_samples = len(dataloader.dataset)
    #print("Number of samples in the dataset:\t{}".format(num_samples))

    training_loss_history = []
    training_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for i in range(epochs):
        print('Epoch {}/{}'.format(i + 1, epochs))
        print('-' * 10)

        running_loss = 0
        running_corrects = 0
        count = 0
        for features, labels in dataloader:

            count += 1
            if (count % 100 == 0):
                print("Completed batch " + str(count))


            # forward pass
            output = model(features)
            loss = criterion(output, labels)
            _, preds = torch.max(output, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()
            running_corrects += int(torch.sum(preds == labels.data))
        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects / num_samples

        print('\tLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        training_acc_history.append(epoch_acc)
        training_loss_history.append(epoch_loss)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc_history, training_loss_history


def hog_predict_shape(model, dataset, original_csv, save_filename = None):
    # takes a csv file from <root_folder> (one that has the cluster predictions) and takes
    # information collected from a shapes_in_clusters csv to predict the shape of an image.  Creates a new
    # csv called shape_predictions where the columns are
    # img path, img class, img shape, shape prediction (string)
    # dataset is a hog_dataset instance

    model.eval()

    if save_filename is None:
        save_filename = "nn_shape_predictions.csv"

    original_csv_df = load_hog_df(original_csv)

    root_folder = os.path.split(original_csv)[0]
    save_path = os.path.join(root_folder, save_filename)
    f = open(save_path, 'w+')
    f.write("img,class,model_predicted_shape,true_shape,correct")

    # array where each index represents a class, and the value at that index represents the attribute label for that
    # class
    class_attributes = split_all("shape")
    mapping = {"circle": 0, "triangle": 1, "diamond": 2, "inverted_triangle": 3, "octagon": 4}
    inv_mapping = {v: k for k, v in mapping.items()}

    total_correct = 0
    shapes_correct = {"circle": 0, "triangle": 0, "diamond": 0, "inverted_triangle": 0, "octagon": 0}
    shapes_count = {"circle": 0, "triangle": 0, "diamond": 0, "inverted_triangle": 0, "octagon": 0}

    num_rows = len(dataset)

    for i in range(num_rows):
        correct = 0

        # get a row of hog data
        data, label = dataset.__getitem__(i)
        label = int(label.item())

        # get model prediction
        pred = list(model(data))
        pred_shape = pred.index(max(pred))

        # get img file name and img class
        data = original_csv_df.iloc[[i], [0,1]]
        data = list(data.iloc[0])
        img_name = data[0]
        true_class = data[1]
        true_shape = class_attributes[true_class]
        if mapping[true_shape] != label:
            print("Error, true class is not equal to the label")
            exit(-1)

        shapes_count[true_shape] +=1
        if pred_shape == mapping[true_shape]:   # correct prediction?
            total_correct +=1
            shapes_correct[true_shape] += 1
            correct = 1

        f.write("\n{},{},{},{},{}".format(img_name, true_class, inv_mapping[pred_shape], true_shape, correct))

    f.close()
    print("total correct:\t\t{}/{}, {:4f}%".format(total_correct, num_rows, 100 * total_correct/num_rows))
    for key in shapes_count:
        print("{}:\t\t{}/{} correct, {:4f}%".format(key, shapes_correct[key],shapes_count[key],
                                             100 * shapes_correct[key]/shapes_count[key]))



def create_train_save_hog_model(filename = None, evaluate = True, save = True):
    global INPUT_SIZE
    INPUT_SIZE = 80
    if filename is None:
        filename = os.path.join(os.getcwd(), "Image_features", "HOG", "80hog_img_features_GTSRB_ResNet_2021-03-11",
                                "80hog_img_features_GTSRB_ResNet_2021-03-11.csv")
    hog_data = hog_dataset(filename)
    print("Number of samples in the dataset:\t{}".format(len(hog_data)))
    train_loader = torch.utils.data.DataLoader(hog_data, batch_size=64, shuffle=True)
    model = Net()
    model, train_acc, train_loss = train_hog_model(model, train_loader)
    if evaluate:
        hog_predict_shape(model, hog_data, original_csv=filename)
    if save:
        fname = os.path.join(os.path.split(filename)[0], str_date() + "hog_model")
        save_model(model, fname)


def load_evaluate_hog(filename = None, test_data = False):
    if filename is None:
        filename = os.path.join(os.getcwd(), "Image_features", "HOG",
                                "80hog_img_features_GTSRB_ResNet_2021-03-11", "2021-04-01hog_model")
    model = load_model(filename)

    csv = os.path.join(os.getcwd(), "Image_features", "HOG", "80hog_img_features_GTSRB_ResNet_2021-03-11",
                                "80hog_img_features_GTSRB_ResNet_2021-03-11.csv")
    if test_data:
        csv = os.path.join(os.getcwd(), "Image_features", "HOG", "80hog_img_features_Test_2021-03-22",
                           "80hog_img_features_Test_2021-03-22.csv")
    hog_data = hog_dataset(csv)
    hog_predict_shape(model, hog_data, original_csv=csv)


if __name__ == '__main__':
    #csv = os.path.join(os.getcwd(), "Image_features", "img_features_GTSRB_ResNet_2020-12-29_normalized.csv")
    #csv = os.path.join(os.getcwd(), "Image_features", "img_features_small_test_dataset_2020-12-29.csv")
    #train(csv)
    #create_train_save_print_model(csv)
    load_evaluate_hog(test_data=True)
    #create_train_save_hog_model(evaluate=False, save=False)