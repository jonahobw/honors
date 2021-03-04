import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import torch.nn.functional as F


def num_grad(f, x):
    delta = 1
    grad = np.zeros(len(x))
    a = np.copy(x)
    print(len(x))
    for i in range(len(x)):
        a[i] = x[i] + delta
        grad[i] = (f(a) - f(x)) / delta
        a[i] -= delta
        print(i)

    return grad


def num_ascent(f, x):
    conf = f(x)
    print("Conf is {}".format(conf))
    count = 0
    while conf < 0.4:
        grad = num_grad(f, x)
        # grad = ndGradient(f)(x)
        print(grad)
        x += grad

        conf = f(x)
        print("Conf {}".format(conf))
    return x


def load_model(filename):
    # load a model from a file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(filename, map_location=device)
    model.eval()
    return model

def preprocess_image(image, path = True, INPUT_SIZE = 224):
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


def save_transform(h, w, x, save_img = None):
    img = x.reshape((h,w, 3)).astype('uint8')
    img = Image.fromarray(img, mode='RGB')
    img.save('output.jpg')
    img = preprocess_image('output.jpg')
    return img


def create_f(h, w, target):
    def f(x, save_img=None):
        pixels = save_transform(h, w, x, save_img)
        output = net(pixels.unsqueeze(dim=0))
        output = F.softmax(output[0], dim=0)
        return output[target].item()
    #return lambda x: f(x, target)
    return f

def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h*w*c).astype('float64')
    #img_array /= 255
    return h, w, img_array

if __name__ == '__main__':
    print("starting")
    net = load_model('pytorch_resnet_saved_11_9_20')
    impath = os.path.join(os.getcwd(), "small_test_dataset","Test", "0", "00000_00002_00001.png")
    img = Image.open(impath)
    h, w, imgarray = linearize_pixels(img)
    target_class = 3

    f = create_f(h, w, target_class)
    print(f(imgarray))
    num_ascent(f, imgarray)


