import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from pytorch_resnet import load_model, test_one_image, get_model_prediction_probs
from scipy.optimize import differential_evolution

# most code in this file adapted from https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/1_one-pixel-attack-cifar10.ipynb

def perturb_image(xs, img_path):
    # given a pertubation tuple xs = (x, y, r, g, b) and an image,
    # change the pixel in the image at location x, y to have value r, g, b

    img = np.array(Image.open(img_path))

    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs


def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    attack_image = perturb_image(xs, img)[0]
    preds = test_one_image(model, attack_image)
    target_class_confidence = preds[target_class]

    # This function should always be minimized, so return its complement if needed
    return target_class_confidence if minimize else 1 - target_class_confidence


def attack_success(xs, img, target_class, model, targeted_attack=False, verbose=False):
    # evaluates the success of an attack.
    # input a perturbed image to the model and get it's prediction vector
    # if this is a targeted attack, return true if the model predicted the image to be of target_class
    # if untargeted, return true if the model's prediction is not the target_class

    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(xs, img)[0]
    preds= test_one_image(model,attack_image)

    target_class_confidence = preds[target_class]
    predicted_class = preds.index(max(preds))
    predicted_class_confidence = preds[predicted_class]

    # If the prediction is what we want (misclassification or targeted classification), return True
    if verbose:
        print('Model Confidence in class  {}:     {:4f}%'.format(target_class, target_class_confidence*100))
        print('Model prediction was class {} with {:4f}% confidence'.format(predicted_class, predicted_class_confidence*100))
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True


def print_image(img, path = True, title = ""):
    # displays an image with title = title
    if path:
        img = mpimg.imread(img)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def attack(img_id, img_class, model, target=None, pixel_count=1, maxiter=75, popsize=400, verbose=False):
    # performs an attack on a model using possible perturbations on 1 input image
    # uses differential evolution to try to find an image that succeeds in fooling the network
    # parameters:
    # img_id     (string):  the file name of the image
    # img_class     (int):  the true class of the image
    # target        (int):  if None, this is not a targeted attack, otherwise, this is the class we want to
    #                           try to get the network to predict
    # pixel_count   (int):  maximum number of pixels allowed in perturbations
    # maxiter       (int):  max number of differential evolution iterations to try before declaring failure
    # popsize       (int):  parameter used for the differential evolution algorithm
    # verbose      (bool):  detailed output if True

    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else img_class

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    im = Image.open(img_id)
    y, x = im.size
    bounds = [(0, x), (0, y), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, img_id, target_class, model, minimize= not targeted_attack)

    def callback_fn(xs, convergence):
        return attack_success(xs, img_id, target_class,
                              model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    prior_probs = test_one_image(model, img_id, path= True)
    prior_true_class_confidence = prior_probs[int(img_class)]

    attack_image = perturb_image(attack_result.x, img_id)[0]
    predicted_probs = test_one_image(model, attack_image)
    predicted_class = predicted_probs.index(max(predicted_probs))
    predicted_class_confidence = predicted_probs[predicted_class]
    true_class_confidence = predicted_probs[int(img_class)]

    success = False
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        success = True
    cdiff = prior_true_class_confidence - true_class_confidence

    # Show the best attempt at a solution (successful or not)

    annotation = 'Model Confidence in class  {}:     {:4f}%'.format(str(img_class), true_class_confidence * 100)
    annotation += '\nModel prediction was class {} with {:4f}% confidence'.format(str(predicted_class), predicted_class_confidence * 100)
    annotation += '\nAttack was {}'.format("successful" if success else "unsuccessful")

    print_image(attack_image, path=False, title = annotation)

    return [model, pixel_count, img_id, img_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]

sign = 0
image_file = os.listdir(os.path.join(os.getcwd(), "Test", str(sign)))[0]
image_file_path = os.path.join(os.getcwd(), "Test", str(sign), image_file)

model = load_model("pytorch_resnet_saved")
attack(image_file_path, sign, model, pixel_count=5, maxiter=15, popsize=50, verbose= True)