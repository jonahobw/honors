from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import sys
import skimage.io

file = os.path.join(os.getcwd(), 'Debug', '00', '03922.png')


def avg_rgb(img_file, plot = True):
    # adapted from https://datacarpentry.org/image-processing/aio/index.html
    image = skimage.io.imread(fname=img_file)

    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    avgs = {}
    r = list(np.concatenate(image[:, :, 0]))
    b = list(np.concatenate(image[:, :, 1]))
    g = list(np.concatenate(image[:, :, 2]))
    avgs["r"] = sum(r) / len(r)
    avgs["g"] = sum(g) / len(g)
    avgs["b"] = sum(b) / len(b)
    print(avgs)

    if plot:
        # create the histogram plot, with three lines, one for
        # each color
        plt.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=c)

        plt.legend()
        plt.xlabel("Color value")
        plt.ylabel("Pixels")

        plt.show()
    return avgs


def get_dominant_colors(image_file, number_of_colors, show_chart):
    # adapted from https://github.com/kb22/Color-Identification-using-Machine-Learning/blob/master/Color%20Identification%20using%20Machine%20Learning.ipynb
    # gets top <number_of_colors> colors from an image using K means clustering

    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    modified_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.pie(counts.values(), colors=hex_colors)
        ax2 = fig.add_subplot(122)
        ax2.imshow(image)
        ax2.axis('off')
        plt.show()

    return rgb_colors


print(get_dominant_colors(file, 8, True))
#print(avg_rgb(file))