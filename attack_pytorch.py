import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from pytorch_resnet import load_model, test_one_image, format_two_digits, NUM_CLASSES
from scipy.optimize import differential_evolution
import random
import time
import multiprocessing
from general import str_date
import logging

#changeable parameters
MODEL_NAME = "pytorch_resnet_saved_11_9_20"
SAMPLES = 1
POP_SIZE = 5
MAX_ITER = 3
PIXELS = [100]
SAVE = True
LOG_LEVEL = logging.DEBUG
TARGETED = False
SHOW_IMAGE = True

#unchangeable parameters
ROOT_SAVE_FOLDER = os.path.join(os.getcwd(), "Outputs", "attacks")
if TARGETED:
    ROOT_SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "targeted")
else:
    ROOT_SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "untargeted")
IMG_FOLDER = ""
PLT_FOLDER = ""
logger = logging.getLogger("attack_log")
logger.setLevel(LOG_LEVEL)

if SAVE:
    save_date = str_date()
    ROOT_SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "{}_{}_{}_samples".format(save_date, MODEL_NAME, str(SAMPLES)))
    os.mkdir(ROOT_SAVE_FOLDER)
    IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "imgs")
    os.mkdir(IMG_FOLDER)
    PLT_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "plots")
    os.mkdir(PLT_FOLDER)
    logfile = os.path.join(ROOT_SAVE_FOLDER, "attack.log")
    logging.basicConfig(filename=logfile, format='%(message)s')
    logging.getLogger("attack_log").addHandler(logging.StreamHandler())
else:
    logging.basicConfig(format='%(message)s')


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
        logger.debug('Model Confidence in true class {}:     {:4f}%'.format(target_class, target_class_confidence*100))
        logger.debug('Model prediction was class     {} with {:4f}% confidence\n'.format(predicted_class, predicted_class_confidence*100))
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True


def print_image(img, path = True, title = ""):
    # displays an image with title = title
    if path:
        img = mpimg.imread(img)
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_perturbed_image(img, title = "", true_class = None, pixels = None, filename =None):
    # saves an image with the filename of <trueclass>_<targetclass>_<date>
    global IMG_FOLDER
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    filename = filename.split(".")[0]
    fname = 'img_{}_class_{}_{}pixels'.format(filename, str(true_class), str(pixels))
    fname = os.path.join(IMG_FOLDER, fname)
    plt.savefig(fname)


def attack(img_id, img_class, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False, show_image = False):
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

    if verbose:
        img_file = os.path.split(img_id)
        logger.debug("---------------Testing image {}---------------".format(img_file[1]))
        prior_probs = test_one_image(model, img_id, path= True)
        prior_true_class_confidence = prior_probs[int(img_class)]
        prior_predicted_class = prior_probs.index(max(prior_probs))
        prior_predicted_class_confidence = prior_probs[prior_predicted_class]
        logger.debug('Prior Model Confidence in true class {}:     {:4f}% '
              '(before attack)'.format(str(img_class), prior_true_class_confidence * 100))
        logger.debug('Prior Model prediction was class     {} with {:4f}% '
              'confidence (before attack)\n'.format(str(prior_predicted_class), 100*prior_predicted_class_confidence))

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
    attack_image = perturb_image(attack_result.x, img_id)[0]
    predicted_probs = test_one_image(model, attack_image)
    predicted_class = predicted_probs.index(max(predicted_probs))
    predicted_class_confidence = predicted_probs[predicted_class]
    true_class_confidence = predicted_probs[int(img_class)]

    success = False
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        success = True
        if verbose:
            logger.debug("Success on image {}\n".format(img_file[1]))
    elif(verbose):
        logger.debug("Reached max iterations, attack unsuccessful on image {}\n".format(img_file[1]))
    #cdiff = prior_true_class_confidence - true_class_confidence

    annotation = 'Model Confidence in true class  {}:     {:4f}%'.format(str(img_class),
                                                                         true_class_confidence * 100)
    annotation += '\nModel prediction was class   {} with {:4f}% confidence'.format(str(predicted_class),
                                                                                    predicted_class_confidence * 100)
    annotation += '\nAttack was {}'.format("successful" if success else "unsuccessful")

    # Show the best attempt at a solution (successful or not)
    if show_image:
        print_image(attack_image, path=False, title = annotation)

    if SAVE and success:
        # saving successfully perturbed images
        save_perturbed_image(attack_image, annotation, img_class, pixel_count, img_file[1])

    return success


def get_correctly_classified_imgs(model, imgs, samples):
    # This function takes an array of image files and returns a subset of that array of size <samples> where
    # every image in the subset is correctly classified by the <model>
    #
    # parameters:
    # model (neural network object): the model to use for classification
    # imgs (list): array of tuples (image file name, image true class)
    # samples (int): the number of images desired who classify correctly

    valid_imgs = []
    for i in range(len(imgs)):
        image, true_class = imgs.pop()
        im = Image.open(image)
        preds = test_one_image(model, im)
        class_pred = preds.index(max(preds))
        if(int(class_pred)==int(true_class)):
            valid_imgs.append((image, true_class))
            samples -= 1
        if(samples == 0):
            break

    if samples > 0:
        logger.error("Error, unable to find {} correctly classified images from provided data".format(str(samples)))
        return -1
    else:
        return valid_imgs


def retrieve_valid_test_images(model, image_folder, samples, targeted = None, exclusive = None):
    # returns a set of images who classify correctly of size <samples>
    #
    # if <targeted> is not none, it should be an integer representing one of the classes, and
    # the function will return a set of images (who classify correctly) of size <samples> where
    # none of the images have true class of <targeted>
    #
    # if <exclusive> is not none, it should be an integer representing one of the classes, and
    # the function returns a set of images (who classify correctly) of size <samples> where
    # all of the images have true class of <exclusive>

    imgs = []

    # <exclusive> case:
    if (exclusive is not None):
        sign_folder = os.path.join(image_folder, format_two_digits(exclusive))
        test_data = os.listdir(sign_folder)
        for i in range(len(test_data)):
            file = test_data[i]
            file_path = os.path.join(sign_folder, file)
            # tuple of (image filename, label of image)
            imgs.append((file_path, exclusive))
        return get_correctly_classified_imgs(model, imgs, samples)

    classes = os.listdir(image_folder)

    for sign in classes:
        if(targeted is not None and int(targeted)==int(sign)):
            continue
        sign_directory = os.path.join(image_folder, str(sign))
        test_data = os.listdir(sign_directory)
        for i in range(len(test_data)):
            file = test_data[i]
            file_path = os.path.join(sign_directory, file)
            # tuple of (image filename, label of image)
            imgs.append((file_path, sign))
    random.shuffle(imgs)
    return get_correctly_classified_imgs(model, imgs, samples)



def attack_all_untargeted(model, image_folder = None, samples=100, pixels=(1, 3, 5),
               maxiter=25, popsize=200, verbose=False, show_image = False):
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:     {}".format(image_folder))
    logger.info("Samples:         {}".format(str(samples)))
    logger.info("Pixels:          {}".format(pixels))
    logger.info("Max iterations:  {}".format(str(maxiter)))
    logger.info("Population size: {}\n\n".format(str(popsize)))

    since = time.time()
    img_samples = retrieve_valid_test_images(model, image_folder, samples)

    # 1d array where index corresponds to pixel count, and the value of an element is the success of
    # an untargeted attack with that pixel count
    results = [0]*len(pixels)

    total_success = 0

    for i, pixel_count in enumerate(pixels):
        logger.info("\n\nAttacking with {} pixels\n".format(pixel_count))
        items_to_remove = []
        for j, (img, label) in enumerate(img_samples):
            if(j%10 == 0 and j != 0):
                logger.info("{} samples tested so far".format(str(j)))
            success = attack(img, int(label), model, pixel_count=pixel_count,
                             maxiter = maxiter, popsize= popsize, verbose=verbose,
                             show_image = show_image)
            if success:
                total_success +=1
                items_to_remove.append(img_samples[j])
        for item in items_to_remove:
            img_samples.remove(item)
        success_percent = 100*total_success/samples
        results[i] = success_percent
        logger.info("Attack success for {}-pixel attack on {} "
              "samples is {:4f}%".format(str(pixel_count), str(samples), success_percent))
        logger.info("{} images were successfully perturbed to trick the model".format(str(total_success)))

    logger.info("Results vector:")
    logger.info(results)
    time_elapsed = time.time() - since
    logger.info('\nAttack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return results, pixels, samples, maxiter, popsize


def attack_all_targeted(model, image_folder = None, samples=500, pixels=(1, 3, 5), num_classes = NUM_CLASSES,
               maxiter=75, popsize=400, verbose=False, show_image = False):
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:     {}".format(image_folder))
    logger.info("Samples:         {}".format(str(samples)))
    logger.info("Pixels:          {}".format(pixels))
    logger.info("Max iterations:  {}".format(str(maxiter)))
    logger.info("Population size: {}\n\n".format(str(popsize)))

    since = time.time()

    test_images = []
    for i in range(NUM_CLASSES):
        test_images.append(retrieve_valid_test_images(model, image_folder, samples, targeted=i))

    # 2d array where 1st dimension is pixel count and 2nd is target class
    results = []
    for i in range(len(pixels)):
        results.append([0]*num_classes)

    for i, pixel_count in enumerate(pixels):
        logger.info("\n\nAttacking with {} pixels\n\n".format(pixel_count))
        for j in range(num_classes):
            total_success = 0
            target_class = j
            img_samples = test_images[j]
            for img, label in img_samples:
                success = attack(img, int(label), model, pixel_count=pixel_count, target= target_class,
                                 maxiter = maxiter, popsize= popsize, verbose=verbose, show_image = show_image)
                if success:
                    total_success +=1
            success_percent = 100 * total_success / samples
            results[i][j] = success_percent
            logger.info("Attack success for {}-pixel attack with target {} on {} "
              "samples is {}\n".format(str(pixel_count), str(target_class), str(samples), str(success_percent)))

    logger.info("\nResults:")
    logger.info(results)
    time_elapsed = time.time() - since
    logger.info('\nAttack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return results, pixels, samples, maxiter, popsize


def attack_targeted(model, target_class, true_class, image_folder = None, samples=500, pixels=(1, 3, 5),
                    maxiter=75, popsize=400, verbose=False, show_image = False):
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:     {}".format(image_folder))
    logger.info("Samples:         {}".format(str(samples)))
    logger.info("Pixels:          {}".format(pixels))
    logger.info("Max iterations:  {}".format(str(maxiter)))
    logger.info("Target class:    {}".format(str(target_class)))
    logger.info("True class:      {}".format(str(true_class)))
    logger.info("Population size: {}\n\n".format(str(popsize)))

    since = time.time()

    img_samples = retrieve_valid_test_images(model, image_folder, samples, exclusive=true_class)

    # 1d array where index corresponds to pixel count, and the value of an element is the success of
    # a targeted attack from <true_class> to <target_class> with that pixel count
    results = [0] * len(pixels)

    total_success = 0

    for i, pixel_count in enumerate(pixels):
        logger.info("\n\nAttacking with {} pixels\n".format(pixel_count))
        items_to_remove = []
        for j, (img, label) in enumerate(img_samples):
            if(j%10 == 0 and j != 0):
                logger.info("{} samples tested so far".format(str(j)))
            success = attack(img, int(label), model, pixel_count=pixel_count, target=target_class,
                             maxiter = maxiter, popsize= popsize, verbose=verbose, show_image = show_image)
            if success:
                total_success +=1
                items_to_remove.append(img_samples[j])
        for item in items_to_remove:
            img_samples.remove(item)
        success_percent = 100*total_success/samples
        results[i] = success_percent
        logger.info("Attack success for {}-pixel attack on {} "
              "samples is {:4f}%".format(str(pixel_count), str(samples), success_percent))
        logger.info("{} images were successfully perturbed to trick the model".format(str(total_success)))

    logger.info("Results vector:")
    logger.info(results)
    time_elapsed = time.time() - since
    logger.info('\nAttack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return results, pixels, samples, maxiter, popsize, target_class, true_class


def plot_untargeted(results, pixels, samples, maxiter, popsize):
    global SAVE, PLT_FOLDER
    plt.figure()
    plt.plot(pixels, results)
    title = 'Attack Success by Number of Pixels, {} samples'.format(str(samples))
    title += "\nMax iterations = {}, Population Size = {}".format(str(maxiter), str(popsize))
    plt.title(title)
    plt.xlabel('Number of Pixels changed')
    plt.xticks(range(max(pixels)+1))
    plt.ylabel('Attack Success (%)')
    if SAVE:
        save_date = str_date()
        fname = "untargeted_{}_samples_{}".format(str(samples), save_date)
        fname = os.path.join(PLT_FOLDER, fname)
        plt.savefig(fname)
    else:
        plt.show()


def plot_targeted(results, pixels, samples, maxiter, popsize, target_class, true_class):
    global SAVE, PLT_FOLDER
    plt.figure()
    plt.plot(pixels, results)
    title = 'Targeted Attack Success by Number of Pixels, {} samples'.format(str(samples))
    title += "\nFrom true class {} to target class {}".format(str(true_class), str(target_class))
    title += "\nMax iterations = {}, Population Size = {}".format(str(maxiter), str(popsize))
    plt.title(title)
    plt.xlabel('Number of Pixels changed')
    plt.xticks(range(max(pixels) + 1))
    plt.ylabel('Attack Success (%)')
    if SAVE:
        save_date = str_date()
        fname = "targeted_{}_samples_{}".format(str(samples), save_date)
        fname = os.path.join(PLT_FOLDER, fname)
        plt.savefig(fname)
    else:
        plt.show()


def setup_saved_folders(model_name, samples):
    global ROOT_SAVE_FOLDER, IMG_FOLDER, PLT_FOLDER
    save_date = str_date()
    ROOT_SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "{}_{}_{}_samples".format(save_date, model_name, str(samples)))
    os.mkdir(ROOT_SAVE_FOLDER)
    IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "imgs")
    os.mkdir(IMG_FOLDER)
    PLT_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "plots")
    os.mkdir(PLT_FOLDER)
    setup_logging()


def setup_logging():
    global ROOT_SAVE_FOLDER, logger
    logfile = os.path.join(ROOT_SAVE_FOLDER, "attack.log")
    logging.basicConfig(filename=logfile)


def run_plot_untargeted():
    globals()
    model = load_model(MODEL_NAME)
    if LOG_LEVEL == logging.DEBUG:
        verbose = True
    else:
        verbose = False
    r, pix, s, m, p = attack_all_untargeted(model, samples=SAMPLES, pixels=PIXELS, maxiter=MAX_ITER,
                                            popsize=POP_SIZE, verbose=verbose, show_image=SHOW_IMAGE)
    plot_untargeted(r, pix, s, m, p)


def run_plot_targeted(target_class, true_class):
    globals()
    model = load_model(MODEL_NAME)
    if LOG_LEVEL == logging.DEBUG:
        verbose = True
    else:
        verbose = False
    res, pix, sam, mxi, pop, tar, tru = attack_targeted(model, samples=SAMPLES, pixels=PIXELS, maxiter=MAX_ITER,
                                                        popsize=POP_SIZE, verbose=verbose, target_class=target_class,
                                                        true_class=true_class, show_image=SHOW_IMAGE)
    plot_targeted(res, pix, sam, mxi, pop, tar, tru)

if __name__ == '__main__':

    starttime = time.time()
    run_plot_untargeted()

    logger.info('That took {} seconds'.format(time.time() - starttime))