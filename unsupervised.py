from sklearn.cluster import KMeans
from image_features import get_hog, unsupervised_predict_one
import os
import shutil
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
from nndt import nndt_depth3_unweighted, nndt_depth4_unweighted, circle_signs, triangle_signs
from attack_helper import spans_multiple_classifiers
from pytorch_resnet import test_one_image, load_model


def load_model_name(name):
    # returns model, nndt
    # nndt is a bool indicating if model is an nndt

    nndt_index = name.find("nndt")

    if name.find("resnet")>=0:
        return load_model(os.path.join(os.getcwd(), "Models", "pytorch_resnet_saved_11_9_20")), False
    elif(nndt_index >= 0):
        name = name[nndt_index:]
        if name.find("3")>=0:
            return nndt_depth3_unweighted(), True
        if name.find("4")>=0:
            return nndt_depth4_unweighted(), True
    else:
        print("Invalid model name in load_model_name, {}".format(name))
        exit(-1)

def setup_logging(folder, name = "flagging"):
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logfile = os.path.join(folder, "{}.log".format(name))
    logging.basicConfig(filename=logfile, format='%(message)s')
    logging.getLogger(name).addHandler(logging.StreamHandler())

def test_get_hog():
    img_file = os.path.join(os.getcwd(), "Train", "00", "00000_00000_00000.png")
    return get_hog(img_file)

def test_unsupervised_pred_one():
    a = test_get_hog()
    b = unsupervised_predict_one(a)
    return b

def test_cluster_pred_vector():
    a = test_unsupervised_pred_one()
    b = cluster_pred_vector(a)
    print(b)

def cluster_pred_vector(pred_vector):
    # takes a prediction vector and uses k-means clustering with k=2 on the prediction vector.
    # this results in 2 sets: classes that are in the prediction and classes that are excluded from the prediction
    pred_vector_np = np.array(pred_vector).reshape((-1, 1))

    kmeans = KMeans(n_clusters=2, random_state=0)
    preds = kmeans.fit_predict(pred_vector_np)
    centers = kmeans.cluster_centers_.reshape(1, -1).tolist()[0]
    included_cluster = centers.index(max(centers))
    included_classes = []
    for i in range(len(preds)):
        if preds[i] == included_cluster:
            included_classes.append(i)
    return included_classes


def flag_one_img(img_path, pred_class, attribute = "HOG"):
    # takes an image and an attribute, gets the feature vector for that attribute, then the image class prediction
    # based on that feature vector, then flags the image if appropriate
    # returns True if image should be flagged, else false

    # FOR HOG, THIS FUNCTION IS 93.8994% CORRECT FOR THE TRAINING DATASET AND 93.4996% FOR THE TEST SET

    # Train set:
    # Avg length of included classes 22.699992348695453, complement is 20.300007651304547

    # Test set:
    #Avg length of included classes 22.869517022961205, complement is 20.130482977038795

    feature_vector = None
    if attribute == "HOG":
        feature_vector = get_hog(img_path)
    else:
        print("Invalide attribute name in flag_imgs, {}".format(attribute))
        exit(-1)

    pred_vect = unsupervised_predict_one(feature_vector)
    included_classes = cluster_pred_vector(pred_vect)
    if pred_class not in included_classes:
        return True
    else:
        return False


def should_be_flagged(true_class, pred_class, attribute = "HOG"):
    # checks to see if adversarial image should have been flagged based on the attribute
    if attribute == "HOG":
        # HOG is based on shape
        shapes = [circle_signs(), triangle_signs(), [12], [13], [14]]
        return spans_multiple_classifiers(shapes, true_class, pred_class)


def flag_imgs_from_attack(attack_folder_name, targeted, tiago = False, attribute = "HOG",
                          log_name = "flagging"):

    # THIS SUCCESSFULLY FLAGGED 16/27 (59.3%) IMAGES THAT SPANNED MULTIPLE SHAPES IN ALL N PIXEL ATTACKS

    model, nndt = load_model_name(attack_folder_name)
    tar = "targeted" if targeted else "untargeted"
    attack_folder = os.path.join(os.getcwd(), "Outputs", "attacks", tar, attack_folder_name)
    img_features_folder = os.path.join(attack_folder, "img_features")
    if not os.path.exists(img_features_folder):
        os.mkdir(img_features_folder)
    img_features_folder = os.path.join(img_features_folder, attribute)
    if os.path.exists(img_features_folder):
        shutil.rmtree(img_features_folder)
    if not os.path.exists(img_features_folder):
        os.mkdir(img_features_folder)
    flagged_folder = os.path.join(img_features_folder, "flagged_imgs")
    os.mkdir(flagged_folder)
    unflagged_folder = os.path.join(img_features_folder, "unflagged_imgs")
    os.mkdir(unflagged_folder)
    setup_logging(img_features_folder, name=log_name)

    img_folder = os.path.join(attack_folder, "raw_imgs")

    # results is a dict where the keys are the number of pixels and the values are a tuple of the form
    # (# of flagged adversarial images for that pixel count, total # of adversarial images for that pixel count)
    results = {}
    total_imgs = 0
    total_flagged = 0
    total_should_flag = 0   # total images that should have been flagged
    total_should_flag_correct = 0   # total images that should have been flagged and were flagged

    pixels = [x for x in os.listdir(img_folder)]
    if tiago:
        pixels = [img_folder]
    for pix_count in pixels:
        pix_folder = os.path.join(img_folder, pix_count)
        adv_imgs = os.listdir(pix_folder)
        adv_img_count = len(adv_imgs)
        total_imgs += adv_img_count
        if tiago:
            logger.info("\nTesting {} images".format(str(adv_img_count)))
        else:
            logger.info("\nTesting {} {} images:".format(str(adv_img_count), str(pix_count)))

        success_count = 0

        for adv_img in adv_imgs:
            should_flag = False
            logger.info("Image {}".format(adv_img))
            adv_im_path = os.path.join(pix_folder, adv_img)
            filename = adv_img.split("_")
            true_class = int(filename[1])

            im = Image.open(adv_im_path)

            if not nndt:
                predicted_probs = test_one_image(model, im)
            else:
                predicted_probs = model.prediction_vector(im, dict=False, path=False)

            predicted_class = predicted_probs.index(max(predicted_probs))
            if should_be_flagged(true_class, predicted_class, attribute):
                total_should_flag += 1
                should_flag = True

            success = flag_one_img(adv_im_path, predicted_class, attribute=attribute)
            if should_flag and success:
                total_should_flag_correct += 1

            annotation = "Image {}".format(adv_img)
            annotation += 'Model prediction was class {}'.format(str(predicted_class))
            annotation += '\n   Flagging {}'.format("successful" if success else "unsuccessful")

            logger.info(annotation + "\n")

            plt.imshow(np.array(im))
            plt.title(annotation)
            plt.tight_layout()

            if success:
                success_count += 1
                fname = os.path.join(flagged_folder, adv_img)
            else:
                fname = os.path.join(unflagged_folder, adv_img)

            plt.savefig(fname)

        results[pix_count] = (success_count, adv_img_count)
        total_flagged += success_count

    if not tiago:
        for result in results:
            transf, total = results[result]
            if total > 0:
                logger.info("\n{} pixels: {}/{} adversarial images flagged, {:4f}%".format(str(result),
                                                                                               str(transf),
                                                                                               str(total),
                                                                                               100 * transf / total))
    if total_imgs > 0:
        logger.info("\n\nTotal: {}/{} images successfully flagged ({:4f}%)".format(str(total_flagged),
                                                                                       str(total_imgs),
                                                                                       100 * total_flagged / total_imgs))
    else:
        logger.info("No successful adversarial images on original attack.")
        return 0, 0, 0, 0

    logger.info("There were {} images which should have been flagged, {} of which were. ({:.4f}%)."
                .format(total_should_flag, total_should_flag_correct,
                        0 if total_should_flag==0 else 100*total_should_flag_correct/total_should_flag))

    return total_should_flag_correct, total_should_flag, total_flagged, total_imgs


def flag_one_attack(tiago = True):
    if tiago:
        flag_imgs_from_attack("2021-04-18_pytorch_resnet_saved_11_9_20_3_imgs_", targeted=True,
                              tiago=tiago)
    else:
        flag_imgs_from_attack("2021-02-02_nndt3_depth3_unweighted_100_samples", targeted=True,
                              tiago=tiago)

def flag_all_N_pixel_attacks(attribute = "HOG"):
    should_be_flagged = 0
    flagged = 0
    total_flagged = 0
    total_imgs = 0

    # untargeted
    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-03-11_pytorch_resnet_saved_11_9_20_100_imgs_",
                                              targeted=False, log_name="flagging_1", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-04-05_nndt3_100_imgs_", targeted=False,
                                              log_name="flagging_2", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-04-05_nndt4_100_imgs_", targeted=False,
                                              log_name="flagging_3", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    # untargeted across nndt3
    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-05_pytorch_resnet_saved_11_9_20_100_imgs_acr_",
                                              targeted=False, log_name="flagging_4", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-05_nndt3_100_imgs_acr", targeted=False,
                                              log_name="flagging_5", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    # untargeted across nndt4
    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-05_pytorch_resnet_saved_11_9_20_100_imgs_acr",
                                              targeted=False, log_name="flagging_6", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-04-14_nndt4_100_imgs_acr", targeted=False,
                                              log_name="flagging_7", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    # targeted
    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-03-11_pytorch_resnet_saved_11_9_20_100_imgs_",
                                              targeted=True, log_name="flagging_9", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-03-22_nndt3_100_imgs_", targeted=True,
                                              log_name="flagging_10", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-03-22_nndt4_100_imgs_", targeted=True,
                                              log_name="flagging_11", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    # targeted across nndt3

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-06_pytorch_resnet_saved_11_9_20_100_imgs_acr",
                                              targeted=True, log_name="flagging_12", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-05_nndt3_100_imgs_acr", targeted=True,
                                              log_name="flagging_13", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    # targeted across nndt4
    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-08_pytorch_resnet_saved_11_9_20_100_imgs_acr",
                                              targeted=True, log_name="flagging_14", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    should_flag, flag, tf, ti = flag_imgs_from_attack("2021-05-06_nndt4_100_imgs_acr", targeted=True,
                                              log_name="flagging_15", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag
    if tf is not None:
        total_flagged += tf
    if ti is not None:
        total_imgs += ti

    print("Attribute {}".format(attribute))
    print("Total: {} images should have been flagged and {} were flagged. ({:.4f}%)"
          .format(should_be_flagged, flagged, 100*flagged/should_be_flagged))
    print("Total {} images were flagged out of {} adversarial images. ({:.4f}%)"
          .format(total_flagged, total_imgs, 100 * total_flagged/total_imgs))


def flag_all_mia_attacks(attribute = "HOG"):
    should_be_flagged = 0
    flagged = 0

    # untargeted
    should_flag, flag = flag_imgs_from_attack("2021-05-08_pytorch_resnet_saved_11_9_20_100_imgs_",
                                              targeted=False, log_name="flagging_1", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt3_100_imgs_", targeted=False,
                                              log_name="flagging_2", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt4_100_imgs_", targeted=False,
                                              log_name="flagging_3", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    # untargeted across nndt3
    should_flag, flag = flag_imgs_from_attack("2021-05-09_pytorch_resnet_saved_11_9_20_100_imgs_acr",
                                              targeted=False, log_name="flagging_4", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt3_100_imgs_acr", targeted=False,
                                              log_name="flagging_5", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    # untargeted across nndt4
    should_flag, flag = flag_imgs_from_attack("2021-05-09_pytorch_resnet_saved_11_9_20_100_imgs_acr_",
                                              targeted=False, log_name="flagging_6", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt4_100_imgs_acr", targeted=False,
                                              log_name="flagging_7", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    # targeted
    should_flag, flag = flag_imgs_from_attack("2021-05-08_pytorch_resnet_saved_11_9_20_100_imgs_",
                                              targeted=True, log_name="flagging_9", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-08_nndt4_100_imgs_", targeted=True,
                                              log_name="flagging_10", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-08_nndt3_100_imgs_", targeted=True,
                                              log_name="flagging_11", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    # targeted across nndt3

    should_flag, flag = flag_imgs_from_attack("2021-05-08_pytorch_resnet_saved_11_9_20_100_imgs_acr_",
                                              targeted=True, log_name="flagging_12", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt3_100_imgs_acr", targeted=True,
                                              log_name="flagging_13", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    # targeted across nndt4
    should_flag, flag = flag_imgs_from_attack("2021-05-08_pytorch_resnet_saved_11_9_20_100_imgs_acr__",
                                              targeted=True, log_name="flagging_14", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    should_flag, flag = flag_imgs_from_attack("2021-05-09_nndt4_100_imgs_acr", targeted=True,
                                              log_name="flagging_15", attribute = attribute)
    if should_flag is not None:
        should_be_flagged += should_flag
    if flag is not None:
        flagged += flag

    print("Attribute {}".format(attribute))
    print("Total: {} images should have been flagged and {} were flagged. ({:.4f}%)"
          .format(should_be_flagged, flagged, 100*flagged/should_be_flagged))

if __name__ == '__main__':
    flag_all_N_pixel_attacks()
    #flag_all_mia_attacks()