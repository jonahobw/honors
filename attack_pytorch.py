import numpy as np
from pytorch_resnet import load_model, test_one_image, NUM_CLASSES, set_cuda
from scipy.optimize import differential_evolution
from nndt import nndt_depth3_unweighted, nndt_depth4_unweighted
import time
from general import str_date
import logging
from attack_helper import *
from attack_parser import *

# the GPU ID to use.  Purpose is for when you want to run multiple attacks simultaneously on different GPUs
GPU_ID = 0

# Whether or not to do a new attack (False) or test the transferability of a completed attack.  If this is a
# transferability test, the model parameters below define the transfer model
TRANSFER = False

# Model parameters
#-----------------------------------------------
# neural net to use.  If this is a regular resnet model, this should be the filename of the model in the
# ./Models folder in string format.  If this is an nndt, this should be the name of the nndt class in string format
MODEL_NAME = "nndt_depth3_unweighted()" #"pytorch_resnet_saved_11_9_20" #
# this should be none if the model is a regular resnet.  If the model is an nndt, then this should be an instance of
# the nndt class
NNDT = nndt_depth3_unweighted()
# If none, no change.  If not none, should be a 2d array where the first dimension are the leaf classifiers of the nndt
# and the second dimension are the classes classified by the leaf classifiers.  This 2d array is obtained by calling the
# static method leaf_classifier_groups() of the NNDT object.  For an untargeted attack: the attack
# will only be called successful if the original class and misclassified class span multiple leaf classifiers.  For a
# targeted attack: attack pairs will be sampled from the set of pairs that span multiple leaf classifiers
ACROSS_CLASSIFIERS = None #nndt_depth4_unweighted.leaf_classifier_groups()

# Attack parameters
#-----------------------------------------------
# Differential Evolution parameters (ints)
POP_SIZE = 1#500
MAX_ITER = 1#30
# Number of pixels to attack (array of ints)
PIXELS = [1]#[1, 3, 5]
# Save into a folder (bool)
SAVE = True
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
# (OPTIONAL, use None if not specifying)
# images to use in attack.  Used if you want to run an attack on the same images for different models.  The format is
# an array of tuples (img path, img class)
UNTAR_IMGS = None

# Targeted attack parameters
#----------------------------------------------
# Number of targeted pairs to attack (int)
ATTACK_PAIRS = 1
# Number of images to attack for each targeted pair (int)
N = 1
# Either attack the pairs with the highest danger weight (False) or random pairs (True)
RANDOM = True
# (OPTIONAL, use None if not specifying)
# images and attack pairs to use in attack.  Used if you want to run an attack on the same images and attack pairs for
# different models.  The format is dict {attack pair: imgs for that attack pair}, where attack pair is a
# tuple of (true class, target class) and imgs for that attack pair is an array of img paths
TAR_IMGS = None

# Testing Transferability Property Parameters
# These parameters are for taking the successful adversarial images from one attack and testing them on another model
# to see if the adversarial images transfer from the attack model to the transfer model
#----------------------------------------------
# name of the root folder of the attack (not a full path, targeted or untargeted is decided by above global variable
# TARGETED)
ATTACK_FOLDER = "2021-02-24_nndt_depth4_unweighted()_100_samples"



def setup_variables():
    globals()
    global logger, IMG_FOLDER, PLT_FOLDER, ROOT_SAVE_FOLDER, MODEL_PATH, NNDT, PIXELS, RAW_IMG_FOLDER, ATTACK_FOLDER, \
        TRANSFER_FOLDER, UNTAR_IMGS, TAR_IMGS, ACROSS_CLASSIFIERS
    if NNDT is None:
        MODEL_PATH = os.path.join(os.getcwd(), "Models", MODEL_NAME)
    logger = logging.getLogger("attack_log")
    logger.setLevel(LOG_LEVEL)

    tar = "targeted" if TARGETED else "untargeted"
    across = "acr" if ACROSS_CLASSIFIERS is not None else ""

    if TRANSFER:
        ATTACK_FOLDER = os.path.join(os.getcwd(), "Outputs", "attacks", tar, ATTACK_FOLDER)

    if SAVE:
        if TRANSFER:
            transferability_dir = os.path.join(ATTACK_FOLDER, "transferability")
            # make transferability folder if it does not exist
            if "transferability" not in os.listdir(ATTACK_FOLDER):
                os.mkdir(transferability_dir)
            TRANSFER_FOLDER = os.path.join(transferability_dir, MODEL_NAME + "_transferability")
            os.mkdir(TRANSFER_FOLDER)
            for pix_count in os.listdir(os.path.join(ATTACK_FOLDER, "raw_imgs")):
                os.mkdir(os.path.join(TRANSFER_FOLDER, pix_count))
            logfile = os.path.join(TRANSFER_FOLDER, "transfer.log")
            logging.basicConfig(filename=logfile, format='%(message)s')
            logging.getLogger("attack_log").addHandler(logging.StreamHandler())
            return

        root_folder_prefix = os.path.join(os.getcwd(), "Outputs", "attacks", tar)
        IMG_FOLDER = ""
        PLT_FOLDER = ""
        save_date = str_date()
        num_images = ATTACK_PAIRS * N if TARGETED else SAMPLES
        ROOT_SAVE_FOLDER = os.path.join(root_folder_prefix, "{}_{}_{}_imgs_{}".format(save_date, MODEL_NAME, str(num_images), across))
        os.mkdir(ROOT_SAVE_FOLDER)
        IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "imgs")
        RAW_IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "raw_imgs")
        os.mkdir(IMG_FOLDER)
        os.mkdir(RAW_IMG_FOLDER)
        for pix_count in PIXELS:
            os.mkdir(os.path.join(IMG_FOLDER, str(pix_count) + "_pixels"))
            os.mkdir(os.path.join(RAW_IMG_FOLDER, str(pix_count) + "_pixels"))
        PLT_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "plots")
        os.mkdir(PLT_FOLDER)
        logfile = os.path.join(ROOT_SAVE_FOLDER, "attack.log")
        logging.basicConfig(filename=logfile, format='%(message)s')
        logging.getLogger("attack_log").addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(format='%(message)s')

    logger.info("Model Name: {}".format(MODEL_NAME))

def setup_variables_cmdline(args):
    globals()
    global logger, IMG_FOLDER, PLT_FOLDER, ROOT_SAVE_FOLDER, MODEL_PATH, NNDT, PIXELS, RAW_IMG_FOLDER, ATTACK_FOLDER, \
        TRANSFER_FOLDER, UNTAR_IMGS, TAR_IMGS, ACROSS_CLASSIFIERS, MODEL_NAME, TARGETED, ATTACK_PAIRS, SAMPLES, \
        N, SAVE, POP_SIZE, MAX_ITER, ACR_NAME

    use_nndt = args.model.find('nndt')>=0 #boolean if model is nndt or not
    if use_nndt:
        if args.model.find('3')>=0:
            NNDT = nndt_depth3_unweighted()
            MODEL_NAME = "nndt3"
        elif args.model.find('4')>=0:
            NNDT = nndt_depth4_unweighted()
            MODEL_NAME = "nndt4"
        else:
            print("Invalid nndt name")
            exit(-1)
    else: #model is not an nndt
        if args.model.find('resnet')>=0:
            args.model = "pytorch_resnet_saved_11_9_20"
        MODEL_PATH = os.path.join(os.getcwd(), "Models", args.model)
        MODEL_NAME = args.model
        NNDT = None

    verbose = args.verbose
    if verbose:
        LOG_LEVEL = logging.DEBUG
    else:
        LOG_LEVEL = logging.INFO

    logger = logging.getLogger("attack_log")
    logger.setLevel(LOG_LEVEL)

    tar = ""
    if args.targeted and not args.untargeted:
        TARGETED = True
        tar = "targeted"
        if args.attack_pairs>0:
            ATTACK_PAIRS = args.attack_pairs
        else:
            print("Attack pairs must be greater than 0")
            exit(-1)
        if args.n>0:
            N = args.n
        else:
            print("Samples per attack pair (N) must be greater than 0")
            exit(-1)
        if args.tar_imgs is not None:
            TAR_IMGS = parse_img_files(args.tar_imgs, True)
            ATTACK_PAIRS = len(list(TAR_IMGS.keys()))
            N = len(list(TAR_IMGS.values())[0])
    elif not args.targeted and args.untargeted:
        TARGETED = False
        tar = "untargeted"
        if args.samples>0:
            SAMPLES = args.samples
        else:
            print("Samples must be greater than 0")
            exit(-1)
        if args.untar_imgs is not None:
            UNTAR_IMGS = parse_img_files(args.untar_imgs, False)
            SAMPLES = len(UNTAR_IMGS)
    else:
        print("Exactly one of -targeted or -untargeted must be specified")
        exit(-1)

    if args.across_classifiers is not None:
        across = "acr"
        if args.across_classifiers.find('3')>=0:
            ACROSS_CLASSIFIERS = nndt_depth3_unweighted.leaf_classifier_groups()
            ACR_NAME = "nndt3"
        elif args.across_classifiers.find('4')>=0:
            ACROSS_CLASSIFIERS = nndt_depth4_unweighted.leaf_classifier_groups()
            ACR_NAME = "nndt4"
        else:
            print("Invalid across_classifiers name")
            exit(-1)
    else:
        ACROSS_CLASSIFIERS = None
        ACR_NAME = "none"
        across = ""

    if args.transfer:
        TRANSFER = True
        ATTACK_FOLDER = os.path.join(os.getcwd(), "Outputs", "attacks", tar, args.attack_folder)
    else:
        TRANSFER = False
        ATTACK_FOLDER = None
        PIXELS = [int(x) for x in args.pixels]

    POP_SIZE = args.pop_size
    MAX_ITER = args.max_iter

    if args.save:
        SAVE = True
        if TRANSFER:
            transferability_dir = os.path.join(ATTACK_FOLDER, "transferability")
            # make transferability folder if it does not exist
            if "transferability" not in os.listdir(ATTACK_FOLDER):
                os.mkdir(transferability_dir)
            TRANSFER_FOLDER = os.path.join(transferability_dir, MODEL_NAME + "_transferability")
            os.mkdir(TRANSFER_FOLDER)
            for pix_count in os.listdir(os.path.join(ATTACK_FOLDER, "raw_imgs")):
                os.mkdir(os.path.join(TRANSFER_FOLDER, pix_count))
            logfile = os.path.join(TRANSFER_FOLDER, "transfer.log")
            logging.basicConfig(filename=logfile, format='%(message)s')
            logging.getLogger("attack_log").addHandler(logging.StreamHandler())
            return

        root_folder_prefix = os.path.join(os.getcwd(), "Outputs", "attacks", tar)
        IMG_FOLDER = ""
        PLT_FOLDER = ""
        save_date = str_date()
        num_images = ATTACK_PAIRS * N if TARGETED else SAMPLES
        ROOT_SAVE_FOLDER = os.path.join(root_folder_prefix,
                                        "{}_{}_{}_imgs_{}".format(save_date, MODEL_NAME, str(num_images), across))
        while(os.path.exists(ROOT_SAVE_FOLDER)):
            ROOT_SAVE_FOLDER += "_"
        os.mkdir(ROOT_SAVE_FOLDER)
        IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "imgs")
        RAW_IMG_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "raw_imgs")
        os.mkdir(IMG_FOLDER)
        os.mkdir(RAW_IMG_FOLDER)
        for pix_count in PIXELS:
            os.mkdir(os.path.join(IMG_FOLDER, str(pix_count) + "_pixels"))
            os.mkdir(os.path.join(RAW_IMG_FOLDER, str(pix_count) + "_pixels"))
        PLT_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "plots")
        os.mkdir(PLT_FOLDER)
        logfile = os.path.join(ROOT_SAVE_FOLDER, "attack.log")
        logging.basicConfig(filename=logfile, format='%(message)s')
        logging.getLogger("attack_log").addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(format='%(message)s')
        SAVE = False

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


def predict_classes(xs, img, target_class, model, minimize=True, nndt = False, gpu_id = None):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    attack_image = perturb_image(xs, img)[0]
    if not nndt:
        preds = test_one_image(model, attack_image, gpu_id=gpu_id)
    else:
        preds = model.prediction_vector(attack_image, dict=False, path=False, gpu_id=gpu_id)
    target_class_confidence = preds[target_class]

    # This function should always be minimized, so return its complement if needed
    return target_class_confidence if minimize else 1 - target_class_confidence


def attack_success(xs, img, img_class, target_class, model, targeted_attack=False, verbose=False, nndt=False,
                   across_classifiers = None, gpu_id = None):
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
        preds= test_one_image(model,attack_image, gpu_id=gpu_id)
    else:
        preds = model.prediction_vector(attack_image, dict=False, path=False, gpu_id=gpu_id)

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
           maxiter=75, popsize=400, verbose=False, show_image = False, across_classifiers = None, gpu_id = None):
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
    # across_classifiers (bool):  only used when nndt is true and this is an untargeted attack (for a targeted attack,
    # this case is handled in the selection of attack pairs) and adds a condition that the untargeted attack only
    # succeeds if it misclassifies an image of class x to x' where x and x' span multiple final classifiers of the nndt

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
        return predict_classes(xs, img_id, target_class, model, minimize= not targeted_attack, nndt=nndt, gpu_id=gpu_id)

    def callback_fn(xs, convergence):
        return attack_success(xs, img_id, img_class, target_class, model, targeted_attack, verbose, nndt=nndt,
                              across_classifiers=across_classifiers, gpu_id=gpu_id)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img_id)[0]
    if not nndt:
        predicted_probs = test_one_image(model, attack_image, gpu_id=gpu_id)
    else:
        predicted_probs = model.prediction_vector(attack_image, dict=False, path=False, gpu_id=gpu_id)

    predicted_class = predicted_probs.index(max(predicted_probs))
    predicted_class_confidence = predicted_probs[predicted_class]
    true_class_confidence = predicted_probs[int(img_class)]
    target_class_confidence = predicted_probs[int(target_class)]

    success = False
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        success = True
        if across_classifiers is not None and not spans_multiple_classifiers(across_classifiers, img_class, predicted_class):
            success = False
    if verbose and success:
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
        save_perturbed_image(attack_image, annotation, img_class, pixel_count, img_file[1], target = target)

    return success


def attack_all_untargeted(model, image_folder = None, samples=100, pixels=(1, 3, 5), across_classifiers = None,
               maxiter=25, popsize=200, verbose=False, show_image = False, nndt = False, untar_imgs = None,
                          gpu_id = None, acr_name = None):
    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:        {}".format(image_folder))
    logger.info("Samples:            {}".format(str(samples)))
    logger.info("Pixels:             {}".format(pixels))
    logger.info("Max iterations:     {}".format(str(maxiter)))
    logger.info("Population size:    {}".format(str(popsize)))
    logger.info("NNDT:               {}".format(nndt))

    if across_classifiers is not None:
        logger.info("Across classifiers: {}\n\n".format(acr_name))
    else:
        logger.info("\n\n")

    since = time.time()
    if untar_imgs is None:
        img_samples = retrieve_valid_test_images(model, image_folder, samples, nndt = nndt, gpu_id=gpu_id)

    #overwrite img samples if imgs have been specified in UNTAR_IMGS global variable
    else:
        img_samples = untar_imgs

    save_img_files(img_samples, False)

    logger.info("IMGS:          {}".format(str(img_samples)))

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
            success = attack(img, int(label), model, pixel_count=pixel_count, across_classifiers=across_classifiers,
                             maxiter = maxiter, popsize= popsize, verbose=verbose, show_image = show_image, nndt = nndt,
                             gpu_id=gpu_id)
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


def attack_all_targeted(model, random = False, image_folder = None, samples_per_pair=500, pixels=(1, 3, 5), maxiter=75,
                        popsize=400, verbose=False, show_image = False, nndt = False, num_attack_pairs = 10,
                        across_classifiers = None, tar_imgs = None, gpu_id = None, acr_name = None):
    # if random = false, attacks the <samples> highest attack pairs by danger weight, else,
    #   chooses attack pairs randomly
    # num_attack_pairs (int) is a global variable that determines how many pairs to attack
    # across_classifiers can only be true if nndt is true, and it filters the sampled attack pairs so that they span
    #   multiple classifiers on the nndt
    # if across_classifiers is not none, acr_name is the name of the nndt for which the across_classifiers represents
    # samples_per_pair (int) determines how many images to attack for each attack pair
    # so the total number of samples is samples_per_pair * num_attack_pairs
    # tar_imgs: optional variable from the global variable TAR_IMGS to specify the attack pairs and imgs before attack
    # gpu_id: integer of the gpu to use (if available)



    if image_folder == None:
        image_folder = os.path.join(os.getcwd(), "Test")

    # get N attack pairs from global variable N
    attack_pairs = retrieve_attack_pairs(num_attack_pairs, random, across_classifiers=across_classifiers)

    # if attack pairs are specified through global variable TAR_IMGS
    if tar_imgs is not None:
        attack_pairs = list(tar_imgs.keys())

    logger.info("-----Attacking Parameters:-----")
    logger.info("Test folder:           {}".format(image_folder))
    logger.info("Pixels:                {}".format(pixels))
    logger.info("Max iterations:        {}".format(str(maxiter)))
    logger.info("Population size:       {}".format(str(popsize)))
    logger.info("# of attack pairs:     {}".format(str(num_attack_pairs)))
    logger.info("Random attack pairs:   {}".format(str(random)))
    logger.info("Samples per pair:      {}".format(str(samples_per_pair)))
    logger.info("Attack pairs:          {}".format(str(attack_pairs)))

    if across_classifiers is not None:
        logger.info("Across classifiers:    {}\n\n".format(acr_name))
    else:
        logger.info("\n\n")

    since = time.time()

    # dict of length <attack pairs> where keys are the attack pair and values are arrays of images tested
    # for this attack pair)
    all_images = {}

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

        if tar_imgs is None:
            img_samples = retrieve_valid_test_images(model, image_folder, samples_per_pair, exclusive=true_class,
                                                     nndt=nndt, gpu_id = gpu_id)
        else:
            img_samps = tar_imgs[(true_class, target_class)]
            img_samples = [(x, true_class) for x in img_samps]



        # bookkeeping for all_images variable
        current_attack_pair = (true_class, target_class)
        current_imgs = [x[0] for x in img_samples]
        all_images[current_attack_pair] = current_imgs
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
                                 nndt = nndt, gpu_id = gpu_id)
                if success:
                    total_success +=1
                    items_to_remove.append(img_samples[j])
            for item in items_to_remove:
                img_samples.remove(item)
            success_percent = 100*total_success/samples_per_pair
            results[i] = success_percent
            logger.info("From true class {} to target class {}:".format(str(true_class), str(target_class)))
            logger.info("Attack success for {}-pixel attack on {} "
                  "samples is {:4f}%".format(str(pixel_count), str(samples_per_pair), success_percent))
            logger.info("{} images were successfully perturbed to trick the model".format(str(total_success)))

        logger.info("From true class {} to target class {}:".format(str(true_class), str(target_class)))
        logger.info("Results vector:")
        logger.info(results)
        all_results[attack_pairs[k]] = results

    save_img_files(all_images, True)


    time_elapsed = time.time() - since
    logger.info("\n\nAll results: ")
    logger.info(all_results)
    logger.info("\n\nIMGs:      {}".format(all_images))
    logger.info('\nAttack complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return all_results, pixels, samples_per_pair, maxiter, popsize


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


def run_plot_attack(targeted = True):
    globals()
    set_cuda(GPU_ID)
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
    if targeted:
        res, pix, sam, mxi, pop = attack_all_targeted(model, random=RANDOM, samples_per_pair=N, pixels=PIXELS,
                                                      maxiter=MAX_ITER, popsize=POP_SIZE, verbose=verbose,
                                                      show_image=SHOW_IMAGE, nndt=nndt, num_attack_pairs=ATTACK_PAIRS,
                                                      across_classifiers=ACROSS_CLASSIFIERS, tar_imgs=TAR_IMGS,
                                                      gpu_id = GPU_ID, acr_name = ACR_NAME)
        plot_targeted(res, pix, mxi, pop)
    else:
        r, pix, s, m, p = attack_all_untargeted(model, samples=SAMPLES, pixels=PIXELS, maxiter=MAX_ITER,
                                                popsize=POP_SIZE, verbose=verbose, show_image=SHOW_IMAGE, nndt=nndt,
                                                across_classifiers=ACROSS_CLASSIFIERS, untar_imgs=UNTAR_IMGS,
                                                gpu_id = GPU_ID, acr_name = ACR_NAME)
        plot_untargeted(r, pix, s, m, p)


def transferability(transfer_model, nndt, img_folder, targeted, across_classifiers, save = True, save_folder = None):
    # parameters:
    # ------------------------------------
    # transfer_model (pytorch object or nndt object):   model that predicts images
    # nndt (boolean):   indicates if transfer_model is an nndt (True) or a pytorch model (False)
    # img_folder (string path): path to the root folder containing raw images that successfully fooled the attack model
    #                       during the attack; this folder is organized into subfolders by pixel count
    # save_folder (string path):    where to save successful adversarial images on the transfer model.  The path to the
    #                               root folder orgainized by subfolders by pixel count
    # targeted (bool): if the attack was targeted or not
    # across_classifiers (2d array):    if not None, then a 2d array where the first dimension are final classifiers in
    #                                   an nndt and the second dimension are the classes in that final classifier,
    #                                   restricts what is called a successful adversarial image in untargeted attacks

    logger.info("-----Attacking Parameters:-----")
    logger.info("Transfer model:        {}".format(str(transfer_model)))
    logger.info("NNDT:                  {}".format("False" if nndt is None else "True"))
    logger.info("Targeted:              {}".format(str(targeted)))
    logger.info("Original images:       {}".format(str(img_folder[-100:])))
    logger.info("Transfer images:       {}".format(str(save_folder[-100:])))
    logger.info("Across Classifers:     {}\n\n".format(str(across_classifiers)))

    # results is a dict where the keys are the number of pixels and the values are a tuple of the form
    # (# of successful transferable adversarial images for that pixel count,
    #  total # of adversarial images for that pixel count)
    results = {}
    total_imgs = 0
    total_transf = 0

    pixels = [x for x in os.listdir(img_folder)]
    for pix_count in pixels:
        pix_folder = os.path.join(img_folder, pix_count)
        adv_imgs = os.listdir(pix_folder)
        adv_img_count = len(adv_imgs)
        total_imgs += adv_img_count
        logger.info("\nTesting {} {} images:".format(str(adv_img_count), str(pix_count)))

        success_count = 0

        for adv_img in adv_imgs:
            logger.info("Image {}".format(adv_img))
            adv_im_path = os.path.join(pix_folder, adv_img)
            filename = adv_img.split("_")
            true_class = int(filename[1])
            target_class = int(filename[3])
            target = target_class if targeted else true_class

            im = Image.open(adv_im_path)

            if not nndt:
                predicted_probs = test_one_image(transfer_model, im)
            else:
                predicted_probs = transfer_model.prediction_vector(im, dict=False, path=False)

            predicted_class = predicted_probs.index(max(predicted_probs))
            predicted_class_confidence = predicted_probs[predicted_class]
            true_class_confidence = predicted_probs[int(true_class)]
            target_class_confidence = predicted_probs[int(target_class)]

            success = False
            if ((targeted and predicted_class == target_class) or
                    (not targeted and predicted_class != target_class)):
                success = True
                if across_classifiers is not None and not spans_multiple_classifiers(across_classifiers,
                                                                                     true_class, predicted_class):
                    success = False

            annotation = '   Model Confidence in true class   {}:     {:4f}%'.format(str(true_class),
                                                                                  true_class_confidence * 100)
            if targeted:
                annotation += '\n   Model confidence in target class {}:     {:4f}%'.format(str(target),
                                                                                         target_class_confidence * 100)
            else:
                annotation += '\n   Model prediction was class    {} with {:4f}% confidence'.format(str(predicted_class),
                                                                                                 predicted_class_confidence * 100)
            annotation += '\n   Transfer {}'.format("successful" if success else "unsuccessful")

            logger.info(annotation + "\n")

            if success:
                success_count +=1
                if save:
                    plt.imshow(np.array(im))
                    plt.title(annotation)
                    plt.tight_layout()
                    fname = os.path.join(save_folder, pix_count, adv_img)
                    plt.savefig(fname)
        results[pix_count] = (success_count, adv_img_count)
        total_transf += success_count

    for result in results:
        transf, total = results[result]
        if total >0:
            logger.info("\n{} pixels: {}/{} adversarial images transferred, {:4f}%".format(str(result),
                                                                                         str(transf),
                                                                                         str(total),
                                                                                         100*transf/total))
    if total_imgs >0:
        logger.info("\n\nTotal: {}/{} images successfully transferred ({:4f}%)".format(str(total_transf),
                                                                                       str(total_imgs),
                                                                                       100*total_transf/total_imgs))
    else:
        logger.info("No successful adversarial images on original attack.")

    return


def run_transfer():
    globals()
    save_folder = TRANSFER_FOLDER if SAVE else None

    if NNDT is None:
        model = load_model(MODEL_PATH)
        nndt = False
    else:
        model = NNDT
        nndt = True

    img_folder = os.path.join(ATTACK_FOLDER, "raw_imgs")
    transferability(transfer_model=model, nndt=nndt, img_folder=img_folder, targeted=TARGETED,
                    across_classifiers=ACROSS_CLASSIFIERS, save=SAVE, save_folder=save_folder)
    return


def save_perturbed_image(img, title = "", true_class = None, pixels = None, filename =None, target = None):
    # saves an image

    # target        (int):  if None, this is an untargeted attack, otherwise, this is the class we want to
    #                           try to get the network to predict

    global IMG_FOLDER, RAW_IMG_FOLDER
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    filename = filename.split(".")[0]
    fname = 'img_{}_class_{}_{}pixels'.format(filename, str(true_class), str(pixels))
    fname = os.path.join(IMG_FOLDER, str(pixels) + "_pixels", fname)
    plt.savefig(fname)

    # also save raw image
    tar = target if target is not None else true_class
    fname = "trueclass_" + str(true_class) + "_target_" + str(tar) + "_" + str(filename)+".png"
    fname = os.path.join(RAW_IMG_FOLDER, str(pixels) + "_pixels", fname)
    im = Image.fromarray(img)
    im.save(fname)

def test_plot():
    im_path = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
    im = Image.open(im_path)
    plt.imshow(np.array(im))
    plt.title("Random title")
    plt.tight_layout()
    fname = os.path.join(os.getcwd(), "DELETE_ME.png")
    plt.savefig(fname)

    im.save(os.path.join(os.getcwd(), "DELETE_ME_2.png"))

def save_img_files(imgs, targeted):
    global ROOT_SAVE_FOLDER, SAVE
    if not SAVE:
        return
    # saves the list of images as "img_files.txt for the attack, so that future attacks can use these same images
    # for targeted = False (untargeted): imgs is an array of tuples (img path, img class) and img_files.txt is stored
    # with imgpath,imgclass on each line

    # for targeted = True (targeted): imgs is a dict of length <attack pairs> where keys are the attack pair
    # and values are arrays of images tested for this attack pair) and img_files.text is stored as
    # startclass,endclass:array_item0,array_item1,array_item2... on each line

    path = os.path.join(ROOT_SAVE_FOLDER, "img_files.txt")
    f = open(path, 'w+')

    if not targeted:
        for i in range(len(imgs)):
            img_path, img_class = imgs[i]
            f.write("{},{}".format(img_path,img_class))
            if i != (len(imgs) - 1):
                f.write('\n')
        f.close()
        return
    else:
        attack_pairs = list(imgs.keys())
        for i in range(len(attack_pairs)):
            start_class, end_class = attack_pairs[i]
            f.write("{},{}::".format(start_class, end_class))
            attack_pair_imgs = imgs[attack_pairs[i]]
            for j in range(len(attack_pair_imgs)):
                f.write("{}".format(attack_pair_imgs[j]))
                if j != (len(attack_pair_imgs) - 1):
                    f.write(",")
            if i!= (len(attack_pairs) - 1):
               f.write('\n')

        f.close()
        return


def parse_img_files(filepath, targeted):
    # parses an img_files.txt file from an attack and retrieves attack imgs
    # for targeted = False (untargeted): returns imgs as an array of tuples (img path, img class) and
    # img_files.txt is stored with imgpath,imgclass on each line

    # for targeted = True (targeted): returns imgs as a dict of length <attack pairs> where keys are the attack pair
    # and values are arrays of images tested for this attack pair) and img_files.text is stored as
    # startclass,endclass:array_item0,array_item1,array_item2... on each line

    if not os.path.exists(filepath):
        print("invalid img_files.txt path, does not exits")
        return(-1)

    f = open(filepath, 'r+')
    lines = [line.rstrip() for line in f]

    if not targeted: #untargeted attack
        imgs = [] # array of tuples (img path, img class)
        for line in lines:
            spl = line.split(",")
            path = spl[0]
            img_class = int(spl[1])
            imgs.append((path, img_class))
    else: # targeted attack
        imgs = {}
        for line in lines:
            colon = line.split("::")
            atk_pair = colon[0].split(",")
            attack_pair = (int(atk_pair[0]), int(atk_pair[1]))
            atk_imgs = colon[1].split(",")
            imgs[attack_pair] = atk_imgs
    f.close()
    return imgs

def run():
    globals()
    starttime = time.time()
    setup_variables()
    if not TRANSFER:
        run_plot_attack(TARGETED)
    else:
        run_transfer()

    logger.info('That took {} seconds'.format(time.time() - starttime))

def run_cmdline():
    parser_obj = getParser()
    args = parser_obj.parse_args()
    print(args)
    setup_variables_cmdline(args)
    if not TRANSFER:
        run_plot_attack(TARGETED)
    else:
        run_transfer()


if __name__ == '__main__':
    run_cmdline()
    #run()
    #test_plot()
    # path = os.path.join(os.getcwd(), "Outputs", "attacks", "targeted",
    #                     "2021-03-11_pytorch_resnet_saved_11_9_20_4_imgs_", "img_files.txt")
    # a = parse_img_files(path, True)
    # print(a)