import argparse
import os.path


def getParser():
    #####
    ##  Main Arguments
    #####

    parser = argparse.ArgumentParser(description='attack arguments')

    parser.add_argument('-model', required=False, default=None,
                        help='Neural net to use.  If this is a regular resnet model, this should be the filename of the'
                             ' model in the ./Models folder in string format. If this is an nndt, this should be the'
                             ' name of the nndt class in string format')

    parser.add_argument('-tiago', required=False, default=False, action='store_true',
                        help='run Tiago\'s black box attack instead of n-pixel attack')

    parser.add_argument('-delta', required=False, default=1, type=int,
                        help='For Tiago\'s attack, the delta to move in the direction of the gradient at each '
                             'iteration')

    parser.add_argument('-epsilon', required=False, default=15, type=int,
                        help='The bound for the L-infinity norm on adversarial images in Tiago\'s attack')

    parser.add_argument('-speedup', required=False, default=2000, type=int,
                        help='For Tiago\'s attack, the number of pixels to change (on average) for each iteration')

    parser.add_argument('-gpu_id', '-gpu', type=int, default=0, required=False,
                        help='The GPU ID to use. Purpose is for when you want to run multiple attacks simultaneously on'
                             ' different GPUs')

    parser.add_argument('-across_classifiers', default=None,
                        help='Name of nndt model to use for across classifiers. For an untargeted attack: the attack '
                             'will only be called successful if the original class and misclassified class span multiple'
                             ' leaf classifiers.  For a targeted attack: attack pairs will be sampled from the set of '
                             'pairs that span multiple leaf classifiers')

    parser.add_argument('-pop_size', '-p', default=500, type=int,
                        help='Differential evolution population size.')

    parser.add_argument('-max_iter', '-iter', default=30, type=int,
                        help='Differential evolution max iterations.')

    parser.add_argument('-pixels', nargs='+', required=False, default=[1, 3, 5], type=int,
                        help='List of pixels to use in attack. (as integers with spaces between them.)')

    parser.add_argument('-save', action='store_true', default=False,
                        help='Save output into a folder.')

    parser.add_argument('-verbose', action='store_true', default=False, required=False,
                        help='More verbose output.')

    parser.add_argument('-show_image', action='store_true', default=False,
                        help='Show each attempt at an adversarial image.')

    parser.add_argument('-targeted', action='store_true', default=False,
                        help='Run targeted attack.')

    parser.add_argument('-untargeted', action='store_true', default=False,
                        help='Run untargeted attack.')

    #----------------- Untargeted Attack Parameters ---------------------
    parser.add_argument('-samples', default=100, type=int,
                        help='For an untargeted attack, number of samples to attack.')

    parser.add_argument('-untar_imgs', default=None, type= str,
                        help='File path to images to use in untargeted attack. Used if you want to run an attack on the '
                             'same images for different models.  The format of the file should be an array of tuples '
                             '(img path, img class) where each tuple is printed on a separate line.')

    # ------------------ Targeted Attack Parameters ----------------------
    parser.add_argument('-attack_pairs', '-pairs', default=100, type=int,
                        help='Number of targeted pairs to attack.')

    parser.add_argument('-n', '-N', default=1, type=int,
                        help='Number of samples per attack pair to attack.')

    parser.add_argument('-tar_imgs', default=None, type=str,
                        help='File path to images to use in targeted attack. Used if you want to run an attack on the '
                             'same images and attack pairs for different models. The format of the file is a dict '
                             '{attack pair: imgs for that attack pair}, where attack pair is a tuple of '
                             '(true class, target class) and imgs for that attack pair is an array of img paths')

    # ------------------ Transferability Parameters ----------------------
    parser.add_argument('-transfer', action='store_true', default=False,
                        help='Whether or not to do a new attack (False) or test the transferability of a completed '
                             'attack.  If this is a transferability test, the model parameters define the transfer '
                             'model')

    parser.add_argument('-transfer_model', type=str,
                        help='Transfer model to use')

    parser.add_argument('-attack_folder', type=str,
                        help='Name of the root folder of the attack (not a full path, targeted or untargeted is decided '
                             'by above argument \'targeted\'.')


    return parser