import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import torch.nn.functional as F
import time
import logging
import random

logger = logging.getLogger("attack_log")


def calculate_new_momentum(grad, prev_grad, momentum, factor = 2):
    # momentum - array of same size as grad and prev_grad indicating how large to step for each value of x
    # if the sign of grad and prev grad is the same, increase momentum by a factor of <factor>.  Else set momentum to 1.
    for i in range(len(grad)):
        if (grad[i] > 0 and prev_grad[i] > 0) or (grad[i] < 0 and prev_grad[i] < 0):
            momentum[i] *= factor
        else:
            momentum[i] = 1
    return momentum


def targeted_num_grad(f, x, prev_grad, delta = 1, momentum = None, speedup = None):
    # momentum - array of the same size as x indicating how large to step for each value of x
    # speedup - if not None, then an integer representing how many pixels to change per iteration, will randomly skip
    #           pixels for each iteration that should average out to <speedup> pixels per iteration
    grad = np.zeros(len(x))
    a = np.copy(x)
    for i in range(len(x)):
        if (i % 2000 == 0):
            logger.debug("  Calculating grad, {} pixels to go".format(str(len(x) - i)))
        if speedup is not None:
            # speed up the attack by introducing random skips that should average out to 100 pixels per iteration
            threshold = speedup/len(x)
            r = random.random()
            if r > threshold:
                continue
        target_pred_x, _, _, _ = f(x)
        a[i] = x[i] + delta
        target_pred_a, _, _, _ = f(a)
        if momentum is None:
            momentum = [1] * len(x)
        else:
            grad[i] = momentum[i] * (target_pred_a - target_pred_x) / delta
        a[i] -= delta

    momentum = calculate_new_momentum(grad, prev_grad, momentum)

    return grad, momentum


def num_ascent(f, x, true_class, target_class, targeted, delta = 1, threshold = 3, max_iter = 100, speedup = 100,
               step_size = 1):
    target_conf, true_conf, pred_conf, pred_class = f(x)

    logger.debug('\nInitial confidences:\nModel Confidence in true class   {}:     {:4f}%'.format(str(true_class),
                                                                            true_conf * 100))
    if targeted:
        logger.debug('Model confidence in target class {}:     {:4f}%'.format(str(target_class),
                                                                                 target_conf * 100))
    else:
        logger.debug('Model prediction was class    {} with {:4f}% confidence'.format(str(pred_class),
                                                                                         pred_conf * 100))


    count = 0
    best_pred_value = None
    prev_conf = target_conf
    momentum = [1] * len(x) # set to 1s
    prev_grad = [0] * len(x) # set to 0s
    for i in range(max_iter):
        if targeted and (pred_class == target_class):
            return True, x
        if not targeted and (pred_class != true_class):
            return True, x
        grad, momentum = targeted_num_grad(f, x, prev_grad, delta = delta, momentum=momentum, speedup=speedup)
        zeroes = np.zeros(len(grad))
        if np.array_equal(grad, zeroes):
            delta *= 1.5
            logger.debug("\nzero gradient, incrementing delta to {}".format(delta))
        # grad = ndGradient(f)(x)
        #print("Grad: {}".format(grad))
        grad = [x*step_size for x in grad]
        if targeted:
            x += grad
        else:
            x -= grad

        target_conf, true_conf, pred_conf, pred_class = f(x)

        logger.debug('\nIteration {}:\nModel Confidence in true class   {}:     {:4f}%'.format(i+1, str(true_class),
                                                                                true_conf * 100))
        if targeted:
            logger.debug('Model confidence in target class {}:     {:4f}%'.format(str(target_class),
                                                                                    target_conf * 100))
        else:
            logger.debug('Model prediction was class    {} with {:4f}% confidence'.format(str(pred_class),
                                                                                            pred_conf * 100))

        if target_conf == prev_conf:
            count +=1
        else:
            count = 0
        if targeted and (best_pred_value is None or target_conf > best_pred_value):
            best_pred_value = target_conf
        elif not targeted and (best_pred_value is None or target_conf < best_pred_value):
            best_pred_value = target_conf
        else:
            count +=1
        if count > threshold:
            logger.debug("confidence not changed for {} iterations, terminating".format(threshold))
            break
        logger.debug("Best confidence so far: {:4f}".format(100*target_conf))
        prev_conf = target_conf
        prev_grad = grad
    return False, x


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
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    return input_tensor

def test_one_image(model, image, path = False):
    # gets the neural network prediction vector for a single image
    model.eval()
    if(isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    image = preprocess_image(image, path = path)
    image = create_batch(image.clone().detach())
    preds = get_model_prediction_probs(model, image)
    return preds

def create_batch(list_of_tensors):
    input_batch = list_of_tensors.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch

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

def save_transform(h, w, x, filname = None):
    img = x.reshape((h,w, 3)).astype('uint8')
    img = Image.fromarray(img, mode='RGB')
    #img.save('output.jpg')
    #img = preprocess_image('output.jpg')
    return img


def create_f(model, h, w, target, true, nndt, gpu_id = 0):
    def f(x):
        #pixels = save_transform(h, w, x, save_img)
        #output = net(pixels.unsqueeze(dim=0))
        # output = F.softmax(output[0], dim=0)
        img = save_transform(h, w, x)
        if not nndt:
            output = test_one_image(model, img)
        else:
            output = model.prediction_vector(img, dict=False, path=False, gpu_id=gpu_id)

        # confidence in target class
        target_pred = output[target]
        # confidence in true class
        true_pred = output[true]
        # model's predicted class (integer)
        pred_conf = max(output)
        model_pred = output.index(pred_conf)
        #return output[target].item()
        return target_pred, true_pred, pred_conf, model_pred
    #return lambda x: f(x, target)
    return f

def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h*w*c).astype('float64')
    #img_array /= 255
    return h, w, img_array

def save_im(img, h, w):
    img = np.reshape(img, (h,w,3))
    print(type(img))
    img = Image.fromarray(img)
    img.save('adversarial.jpg')

def attack_one(model, img_path, trueclass, targetclass, targeted, nndt = False, delta = 5, max_iter = 100, gpu_id = 0,
               speedup = 100):
    logger.info("---------------Testing image {}---------------".format(img_path))
    img = Image.open(img_path)
    h, w, imgarray = linearize_pixels(img)
    f = create_f(model, h, w, targetclass, trueclass, nndt, gpu_id=gpu_id)
    success, x = num_ascent(f, imgarray, trueclass, targetclass, targeted, delta=delta, max_iter=max_iter,
                            speedup=speedup)

    logger.info("Attack {}".format("successful" if success else "unsuccessful"))

    target_conf, true_conf, pred_conf, model_pred = f(x)

    annotation = 'Model Confidence in true class   {}:     {:4f}%'.format(str(trueclass),
                                                                          true_conf * 100)
    if targeted:
        annotation += '\nModel confidence in target class {}:     {:4f}%'.format(str(targetclass),
                                                                                 target_conf * 100)
    else:
        annotation += '\nModel prediction was class    {} with {:4f}% confidence'.format(str(model_pred),
                                                                                         pred_conf * 100)
    annotation += '\nAttack was {}\n'.format("successful" if success else "unsuccessful")

    img = x.reshape((h,w, 3)).astype('uint8')
    img = Image.fromarray(img, mode='RGB')
    return success, img, annotation


if __name__ == '__main__':
    print("starting")
    since = time.time()
    net = load_model(os.path.join(os.getcwd(), "Models", 'pytorch_resnet_saved_11_9_20'))
    impath = os.path.join(os.getcwd(), "small_test_dataset","Test", "0", "00000_00002_00001.png")
    img = Image.open(impath)
    h, w, imgarray = linearize_pixels(img)
    target_class = 0
    delta = 10

    f = create_f(h, w, target_class)
    print(f(imgarray))
    x = num_ascent(f, imgarray)
    time_elapsed = time.time() - since
    print('Attack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    save_transform(h, w, x)



