import numpy as np
from pytorch_resnet import load_model, test_one_image, NUM_CLASSES
from scipy.optimize import differential_evolution
from nndt import nndt_depth3_unweighted
import time
from general import str_date
import logging
from attack_helper import *

# Model parameters
#-----------------------------------------------
# neural net to use.  If this is a regular resnet model, this should be the filename of the model in the
# ./Models folder in string format.  If this is an nndt, this should be the name of the nndt class in string format
MODEL_NAME = "nndt_depth3_unweighted"
# this should be none if the model is a regular resnet.  If the model is an nndt, then this should be an instance of
# the nndt class
NNDT = nndt_depth3_unweighted()
# If none, no change.  If not none, should be a 2d array where the first dimension are the leaf classifiers of the nndt
# and the second dimension are the classes classified by the leaf classifiers.  For an untargeted attack: the attack
# will only be called successful if the original class and misclassified class span multiple leaf classifiers.  For a
# targeted attack: attack pairs will be sampled from the set of pairs that span multiple leaf classifiers
ACROSS_CLASSIFIERS = None

# Attack parameters
#-----------------------------------------------
# Differential Evolution parameters (ints)
POP_SIZE = 500
MAX_ITER = 50
# Number of pixels to attack (array of ints)
PIXELS = [5]
# Save into a folder (bool)
SAVE = False
# Verbose output (logging.DEBUG for verbose, else logging.INFO)
LOG_LEVEL = logging.DEBUG
# Show each attempt at an adversarial image (bool)
SHOW_IMAGE = False
# Targeted attack (bool)
TARGETED = False

# Untargeted attack parameters
#----------------------------------------------
# number of pictures to attack (int)
SAMPLES = 1

# Targeted attack parameters
#----------------------------------------------
# Number of targeted pairs to attack (int)
ATTACK_PAIRS = 1
# Number of images to attack for each targeted pair (int)
N = 3
# Either attack the pairs with the highest danger weight (False) or random pairs (True)
RANDOM = True


def setup_variables():
    globals()
    global logger, IMG_FOLDER, PLT_FOLDER, ROOT_SAVE_FOLDER, MODEL_PATH, NNDT, PIXELS
    if NNDT is None:
        MODEL_PATH = os.path.join(os.getcwd(), "Models", MODEL_NAME)
    logger = logging.getLogger("attack_log")
    logger.setLevel(LOG_LEVEL)

    if SAVE:
        # unchangeable parameters
        tar = "targeted" if TARGETED else "untargeted"
        root_folder_prefix = os.path.join(os.getcwd(), "Outputs", "attacks", tar)
        IMG_FOLDER = ""
        PLT_FOLDER = ""
        save_date = str_date()
        num_images = ATTACK_PAIRS * N if TARGETED else SAMPLES
        ROOT_SAVE_FOLDER = os.path.join(root_folder_prefix, "{}_{}_{}_samples".format(save_date, MODEL_NAME, str(num_images)))
        os.mkdir(ROOT_SAVE_FOLDER)
        IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "imgs")
        os.mkdir(IMG_FOLDER)
        for pix_count in PIXELS:
            os.mkdir(os.path.join(IMG_FOLDER, str(pix_count) + "_pixels"))
        PLT_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "plots")
        os.mkdir(PLT_FOLDER)
        logfile = os.path.join(ROOT_SAVE_FOLDER, "attack.log")
        logging.basicConfig(filename=logfile, format='%(message)s')
        logging.getLogger("attack_log").addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(format='%(message)s')

    logger.info("Model Name: {}".format(MODEL_NAME))


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


def predict_classes(xs, img, target_class, model, minimize=True, nndt = False):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    attack_image = perturb_image(xs, img)[0]
    if not nndt:
        preds = test_one_image(model, attack_image)
    else:
        preds = model.prediction_vector(attack_image, dict=False, path=False)
    target_class_confidence = preds[target_class]

    # This function should always be minimized, so return its complement if needed
    return target_class_confidence if minimize else 1 - target_class_confidence


def attack_success(xs, img, img_class, target_class, model, targeted_attack=False, verbose=False, nndt=False,
                   across_classifiers = None):
    # evaluates the success of an attack.
    # input a perturbed image to the model and get it's prediction vector
    # if this is a targeted attack, return true if the model predicted the image to be of target_class
    # if untargeted, return true if the model's prediction is not the target_class

    # across_classifiers: If none, no change.  If not none, should be a 2d array where the first dimension are the leaf
    # classifiers of the nndt and the second dimension are the classes classified by the leaf classifiers.  The function
    # will only return true if the predicted class and true class span multiple classifiers

    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(xs, img)[0]
    if not nndt:
        preds= test_one_image(model,attack_image)
    else:
        preds = model.prediction_vector(attack_image, dict=False, path=False)

    target_class_confidence = preds[target_class]
    predicted_class = preds.index(max(preds))
    predicted_class_confidence = preds[predicted_class]
    true_class_confidence = preds[int(img_class)]

    # If the prediction is what we want (misclassification or targeted classification), return True
    if verbose:
        annotation = '\nModel Confidence in true class   {}:     {:4f}%'.format(str(img_class),
                                                                              true_class_confidence * 100)
        if targeted_attack:
            annotation += '\nModel confidence in target class {}:     {:4f}%'.format(str(target_class),
                                                                                     target_class_confidence * 100)
        else:
            annotation += '\nModel prediction was class    {} with {:4f}% confidence'.format(str(predicted_class),
                                                                                             predicted_class_confidence * 100)
        logger.debug(annotation)
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        if(across_classifiers is not None and not spans_multiple_classifiers(across_classifiers, int(img_class), int(predicted_class))):
            return
        return True


def attack(img_id, img_class, model, target=None, pixel_count=1, nndt=False,
           maxiter=75, popsize=400, verbose=False, show_image = False):
    # performs an attack on a model using possible perturbations on 1 input image
    # uses differential evolution to try to find an image that succeeds in fooling the network
    # parameters:
    # img_id     (string):  the file name of the image
    # img_class     (int):  the true class of the image
    # target        (int):  if None, this is an untargeted attack, otherwise, this is the class we want to
    #                           try to get the network to predict
    # pixel_count   (int):  maximum number of pixels allowed in perturbations
    # maxiter       (int):  max number of differential evolution iterations to try before declaring failure
    # popsize       (int):  parameter used for the differential evolution algorithm
    # verbose      (bool):  detailed output if True
    # nndt         (bool):  indicates whether or not the model is an nndt

    if verbose:
        img_file = os.path.split(img_id)
        logger.debug("---------------Testing image {}---------------".format(img_file[1]))
        if not nndt:
            prior_probs = test_one_image(model, img_id, path= True)
        else:
            prior_probs = model.prediction_vector(img_id, dict=False)
        prior_true_class_confidence = prior_probs[int(img_class)]
        logger.debug('Prior Model Confidence in true class {}:     {:4f}% '
              '(before attack)'.format(str(img_class), prior_true_class_confidence * 100))

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
        return predict_classes(xs, img_id, target_class, model, minimize= not targeted_attack, nndt=nndt)

    def callback_fn(xs, convergence):
        return attack_success(xs, img_id, img_class, target_class,
                              model, targeted_attack, verbose, nndt=nndt)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img_id)[0]
    if not nndt:
        predicted_probs = test_one_image(model, attack_image)
    else:
        predicted_probs = model.prediction_vector(attack_image, dict=False, path=False)

    predicted_class = predicted_probs.index(max(predicted_probs))
    predicted_class_confidence = predicted_probs[predicted_class]
    true_class_confidence = predicted_probs[int(img_class)]
    target_class_confidence = predicted_probs[int(target_class)]

    success = False
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        success = True
        if verbose:
            logger.debug("Success on image {}\n".format(img_file[1]))
    elif(verbose):
        logger.debug("Reached max iterations, attack unsuccessful on image {}\n".format(img_file[1]))

    annotation = 'Model Confidence in true class   {}:     {:4f}%'.format(str(img_class),
                                                                         true_class_confidence * 100)
    if targeted_attack:
        annotation += '\nModel confidence in target class {}:     {:4f}%'.format(str(target),
                                                                                 target_class_confidence * 100)
    else:
        annotation += '\nModel prediction was class    {} with {:4f}% confidence'.format(str(predicted_class),
                                                                                    predicted_class_confidence * 100)
    annotation += '\nAttack was {}'.format("successful" if success else "unsuccessful")

    # Show the best attempt at a solution (successful or not)
    if show_image:
        print_image(attack_image, path=False, title = annotation)

    if SAVE and success:
        # saving successfully perturbed images
        save_perturbed_image(attack_image, annotation, img_class, pixel_count, img_file[1])

    return success


def attack_all_untargeted(model, image_folder = None, samples=100, pixels=(1, 3, 5),
               maxiter=25, popsize=200, verbose=False, show_image = False, nndt = False):
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:     {}".format(image_folder))
    logger.info("Samples:         {}".format(str(samples)))
    logger.info("Pixels:          {}".format(pixels))
    logger.info("Max iterations:  {}".format(str(maxiter)))
    logger.info("Population size: {}\n\n".format(str(popsize)))

    since = time.time()
    img_samples = retrieve_valid_test_images(model, image_folder, samples, nndt = nndt)

    # 1d array where index corresponds to pixel count, and the value of an element is the success of
    # an untargeted attack with that pixel count
    results = [0]*len(pixels)

    total_success = 0

    for i, pixel_count in enumerate(pixels):
        logger.info("\n\nAttacking with {} pixels\n".format(pixel_count))
        items_to_remove = []
        for j, (img, label) in enumerate(img_samples):
            logger.debug("Image {}".format(str(j+1)))
            if((j+1)%10 == 0 and (j+1) != 0):
                logger.info("{} samples tested so far".format(str(j)))
            success = attack(img, int(label), model, pixel_count=pixel_count,
                             maxiter = maxiter, popsize= popsize, verbose=verbose,
                             show_image = show_image, nndt = nndt)
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


def attack_all_targeted(model, random = False, image_folder = None, samples=500, pixels=(1, 3, 5),
                    maxiter=75, popsize=400, verbose=False, show_image = False, nndt = False):
    # attacks the N highest attack pairs by danger weight
    # ATTACK_PAIRS (int) is a global variable that determines how many pairs to attack
    # N (int) is a global variable that determines how many images to attack for each attack pair
    # so the total number of samples is N * ATTACK_PAIRS

    globals()
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    # get N attack pairs from global variable N
    attack_pairs = retrieve_attack_pairs(ATTACK_PAIRS, random)

    logger.info("-----Attacking Parameters:-----")
    logger.info("Random attack pairs:   {}".format(str(random)))
    logger.info("Test folder:           {}".format(image_folder))
    logger.info("Pixels:                {}".format(pixels))
    logger.info("Max iterations:        {}".format(str(maxiter)))
    logger.info("Population size:       {}".format(str(popsize)))
    logger.info("# of attack pairs:     {}".format(str(ATTACK_PAIRS)))
    logger.info("Samples per pair:      {}".format(str(N)))
    logger.info("Attack pairs:          {}\n\n".format(str(attack_pairs)))


    since = time.time()

    # all_results is a dict of the form <attack pair> : <results of attack>
    # where <attack pair> is of the form (true_class, target_class)
    # and <results of attack> is an array of length len(pixels) that shows the success of each n-pixel targeted
    # attack from true_class to target_class
    # example:
    # (5, 27) : [0.5, 0.6]
    # if pixels = [1, 3], and N = 10, then the above result means that 50% of the 10 images of true class 5 were
    # able to be perturbed to successfully fool the model as class 27 for a 1 pixel attack, and 60% for a 3 pixel
    # attack
    all_results = {}

    # loop over set of attack pairs
    for k, (true_class, target_class) in enumerate(attack_pairs):
        img_samples = retrieve_valid_test_images(model, image_folder, N, exclusive=true_class, nndt=nndt)
        logger.info("\nTargeted Attack from True Class {} to Target Class {}\n".format(str(true_class), str(target_class)))

        # 1d array where index corresponds to pixel count, and the value of an element is the success of
        # a targeted attack from <true_class> to <target_class> with that pixel count
        results = [0] * len(pixels)

        total_success = 0

        for i, pixel_count in enumerate(pixels):
            logger.info("\n\nAttacking with {} pixels\n".format(pixel_count))
            items_to_remove = []
            for j, (img, label) in enumerate(img_samples):
                logger.debug("Image {}".format(str(j + 1)))
                success = attack(img, int(label), model, pixel_count=pixel_count, target=target_class,
                                 maxiter = maxiter, popsize= popsize, verbose=verbose, show_image = show_image,
                                 nndt = nndt)
                if success:
                    total_success +=1
                    items_to_remove.append(img_samples[j])
            for item in items_to_remove:
                img_samples.remove(item)
            success_percent = 100*total_success/samples
            results[i] = success_percent
            logger.info("From true class {} to target class {}:".format(str(true_class), str(target_class)))
            logger.info("Attack success for {}-pixel attack on {} "
                  "samples is {:4f}%".format(str(pixel_count), str(samples), success_percent))
            logger.info("{} images were successfully perturbed to trick the model".format(str(total_success)))

        logger.info("From true class {} to target class {}:".format(str(true_class), str(target_class)))
        logger.info("Results vector:")
        logger.info(results)
        all_results[attack_pairs[k]] = results

    time_elapsed = time.time() - since
    logger.info("\n\nAll results: ")
    logger.info(all_results)
    logger.info('\nAttack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return all_results, pixels, samples, maxiter, popsize


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


def plot_targeted(results, pixels, maxiter, popsize):
    global SAVE, PLT_FOLDER, ATTACK_PAIRS, N
    samples = ATTACK_PAIRS * N

    # adapted from https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    pixel_results = {}
    for pix_count in pixels:
        pixel_results[pix_count] = []
    labels = []

    for key in results:
        labels.append(str(key))
        for i, percent in enumerate(results[key]):
            pixel_results[pixels[i]].append(percent)

    x = np.arange(len(labels))
    width = 0.75
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.xlabel("Attack Pair")

    offset = 0
    for i in pixels:
        rect = ax.bar(x + offset, pixel_results[i], width / (len(pixels)), label='{} pixels'.format(str(i)))
        autolabel(rect)
        offset += width / (len(pixels))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Attack Success')
    plt.ylim(0, 1)
    ax.set_xticks(x + width / 2 - width / (2 * len(pixels)))
    ax.set_xticklabels(labels)
    ax.legend()

    title = 'Targeted Attack Success by Attack Pair and Number of Pixels, {} samples'.format(str(samples))
    title += "\nMax iterations = {}, Population Size = {}".format(str(maxiter), str(popsize))
    ax.set_title(title)

    if SAVE:
        save_date = str_date()
        fname = "targeted_{}_samples_{}".format(str(samples), save_date)
        fname = os.path.join(PLT_FOLDER, fname)
        plt.savefig(fname)
    else:
        plt.show()


def run_plot_untargeted():
    globals()
    if NNDT is None:
        model = load_model(MODEL_PATH)
        nndt = False
    else:
        model = NNDT
        nndt = True
    if LOG_LEVEL == logging.DEBUG:
        verbose = True
    else:
        verbose = False
    r, pix, s, m, p = attack_all_untargeted(model, samples=SAMPLES, pixels=PIXELS, maxiter=MAX_ITER,
                                            popsize=POP_SIZE, verbose=verbose, show_image=SHOW_IMAGE, nndt=nndt)
    plot_untargeted(r, pix, s, m, p)


def run_plot_targeted():
    globals()
    if NNDT is None:
        model = load_model(MODEL_PATH)
        nndt = False
    else:
        model = NNDT
        nndt = True
    if LOG_LEVEL == logging.DEBUG:
        verbose = True
    else:
        verbose = False
    res, pix, sam, mxi, pop = attack_all_targeted(model, random = RANDOM, samples=N, pixels=PIXELS,
                                                       maxiter=MAX_ITER, popsize=POP_SIZE, verbose=verbose,
                                                       show_image=SHOW_IMAGE, nndt = nndt)
    plot_targeted(res, pix, mxi, pop)


def save_perturbed_image(img, title = "", true_class = None, pixels = None, filename =None):
    # saves an image
    global IMG_FOLDER
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    filename = filename.split(".")[0]
    fname = 'img_{}_class_{}_{}pixels'.format(filename, str(true_class), str(pixels))
    fname = os.path.join(IMG_FOLDER, str(pixels) + "_pixels", fname)
    plt.savefig(fname)


if __name__ == '__main__':
    globals()
    starttime = time.time()
    setup_variables()
    if TARGETED:
        run_plot_targeted()
    else:
        run_plot_untargeted()

    logger.info('That took {} seconds'.format(time.time() - starttime))