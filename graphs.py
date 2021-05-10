import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('ggplot')

def generate_MIA_graphs():
    labels = ["argeted Attack 1", "argeted Attack 2", "argeted Attack 3"]
    rn = "ResNet18"
    n3 = "NNDT3"
    n4 = "NNDT4"
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    n3_color = '#ff7f0e'
    n4_color = '#2ca02c'
    legend_size = 8

    fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize = (5, 8))


    # untargeted attacks ---------------------------------------------------------------------

    untar_labels = ["Unt"+x for x in labels]
    print(untar_labels)
    resnet = [83, 13, 20]
    nndt3 = [86, 7, 0]
    nndt4 = [84, 0, 10]

    rects1 = axs[0].bar(x - width, resnet, width, label=rn)
    rects2 = axs[0].bar(x, nndt3, width, label=n3)
    rects3 = axs[0].bar(x + width, nndt4, width, label=n4)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_ylabel('Successful\nAdversarial Images')
    axs[0].set_xlabel('Attack Scenario')
    axs[0].set_title('Untargeted Attacks')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(untar_labels)
    axs[0].legend(loc='upper right', prop = {'size': legend_size})
    resnet.extend(nndt3)
    resnet.extend(nndt4)
    axs[0].set_ylim([0, int(max(resnet)*1.2)])

    axs[0].bar_label(rects1, padding=3)
    axs[0].bar_label(rects2, padding=3)
    axs[0].bar_label(rects3, padding=3)

    axs[0].set_title('Untargeted Attacks')

    # targeted attacks ---------------------------------------------------------------------
    tar_labels = ["T" + x for x in labels]
    resnet = [59, 52, 58]
    nndt3 = [53, 41, 0]
    nndt4 = [49, 0, 51]

    rects1 = axs[1].bar(x - width, resnet, width, label=rn)
    rects2 = axs[1].bar(x, nndt3, width, label=n3)
    rects3 = axs[1].bar(x + width, nndt4, width, label=n4)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1].set_ylabel('Successful\nAdversarial Images')
    axs[1].set_xlabel('Attack Scenario')
    axs[1].set_title('Untargeted Attack 1')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(tar_labels)
    axs[1].legend(loc='upper right', prop = {'size': legend_size})
    resnet.extend(nndt3)
    axs[1].set_ylim([0, int(max(resnet)*1.2)])

    axs[1].bar_label(rects1, padding=3)
    axs[1].bar_label(rects2, padding=3)
    axs[1].bar_label(rects3, padding=3)

    axs[1].set_title('Targeted Attacks')

    plt.suptitle("Model Inversion Attack Results")
    plt.show()

def generate_n_pixel_graphs():
    labels = ["1", "3", "5"]
    rn = "ResNet18"
    n3 = "NNDT3"
    n4 = "NNDT4"
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    n3_color = '#ff7f0e'
    n4_color = '#2ca02c'
    legend_size = 8

    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True, figsize = (5, 8))


    # untargeted attack 1 ---------------------------------------------------------------------

    resnet = [19, 42, 44]
    nndt3 = [14, 33, 36]
    nndt4 = [13, 32, 35]

    rects1 = axs[0, 0].bar(x - width, resnet, width, label=rn)
    rects3 = axs[0, 0].bar(x, nndt3, width, label=n3)
    rects2 = axs[0, 0].bar(x + width, nndt4, width, label=n4)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0, 0].set_ylabel('Successful\nAdversarial Images')
    axs[0, 0].set_xlabel('Number of Pixels Attacked')
    axs[0, 0].set_title('Untargeted Attack 1')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    resnet.extend(nndt4)
    axs[0, 0].set_ylim([0, int(max(resnet)*1.4)])

    axs[0, 0].bar_label(rects1, padding=3)
    axs[0, 0].bar_label(rects2, padding=3)
    axs[0, 0].bar_label(rects3, padding=3)

    axs[0, 0].set_title('Untargeted Attack 1')

    # untargeted attack 2 ---------------------------------------------------------------------

    resnet = [1, 2, 4]
    nndt3 = [0, 3, 3]

    rects1 = axs[1, 0].bar(x - width / 2, resnet, width, label=rn)
    rects2 = axs[1, 0].bar(x + width / 2, nndt3, width, label=n3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1, 0].set_ylabel('Successful\nAdversarial Images')
    axs[1, 0].set_xlabel('Number of Pixels Attacked')
    axs[1, 0].set_title('Untargeted Attack 1')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    axs[1, 0].set_ylim([0, int(max(resnet)*1.4)])

    axs[1, 0].bar_label(rects1, padding=3)
    axs[1, 0].bar_label(rects2, padding=3)

    axs[1, 0].set_title('Untargeted Attack 2\nSpanning NNDT3 Classifiers')

    # untargeted attack 3 ---------------------------------------------------------------------

    resnet = [3, 9, 11]
    nndt4 = [3, 5, 6]

    rects1 = axs[2, 0].bar(x - width / 2, resnet, width, label=rn)
    rects2 = axs[2, 0].bar(x + width / 2, nndt4, width, label=n4, color = n4_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[2, 0].set_ylabel('Successful\nAdversarial Images')
    axs[2, 0].set_xlabel('Number of Pixels Attacked')
    axs[2, 0].set_title('Untargeted Attack 1')
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels(labels)
    axs[2, 0].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    axs[2, 0].set_ylim([0, int(max(resnet)*1.4)])

    axs[2, 0].bar_label(rects1, padding=3)
    axs[2, 0].bar_label(rects2, padding=3)

    axs[2, 0].set_title('Untargeted Attack 3\nSpanning NNDT4 Classifiers')

    # targeted attack 1 ---------------------------------------------------------------------

    resnet = [0, 2, 4]
    nndt3 = [0, 0, 0]
    nndt4 = [0, 1, 1]

    rects1 = axs[0, 1].bar(x - width, resnet, width, label=rn)
    rects2 = axs[0, 1].bar(x, nndt3, width, label=n3)
    rects3 = axs[0, 1].bar(x + width, nndt4, width, label = n4)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #axs[0, 1].set_ylabel('Successful Adversarial Images')
    axs[0, 1].set_xlabel('Number of Pixels Attacked')
    axs[0, 1].set_title('Untargeted Attack 1')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    resnet.extend(nndt4)
    axs[0, 1].set_ylim([0, int(max(resnet)*1.4)])

    axs[0, 1].bar_label(rects1, padding=3)
    axs[0, 1].bar_label(rects2, padding=3)
    axs[0, 1].bar_label(rects3, padding=3)

    axs[0, 1].set_title('Targeted Attack 1')

    # targeted attack 2 ---------------------------------------------------------------------

    resnet = [0, 1, 4]
    nndt3 = [0, 0, 0]

    rects1 = axs[1, 1].bar(x - width / 2, resnet, width, label=rn)
    rects2 = axs[1, 1].bar(x + width / 2, nndt3, width, label=n3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #axs[1, 1].set_ylabel('Successful Adversarial Images')
    axs[1, 1].set_xlabel('Number of Pixels Attacked')
    axs[1, 1].set_title('Untargeted Attack 1')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    axs[1, 1].set_ylim([0, int(max(resnet)*1.4)])

    axs[1, 1].bar_label(rects1, padding=3)
    axs[1, 1].bar_label(rects2, padding=3)

    axs[1, 1].set_title('Targeted Attack 2\nSpanning NNDT3 Classifiers')

    # targeted attack 3 ---------------------------------------------------------------------

    resnet = [0, 2, 3]
    nndt4 = [0, 0, 0]

    rects1 = axs[2, 1].bar(x - width / 2, resnet, width, label=rn)
    rects2 = axs[2, 1].bar(x + width / 2, nndt4, width, label=n4, color = n4_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #axs[2, 1].set_ylabel('Successful Adversarial Images')
    axs[2, 1].set_xlabel('Number of Pixels Attacked')
    axs[2, 1].set_title('Untargeted Attack 1')
    axs[2, 1].set_xticks(x)
    axs[2, 1].set_xticklabels(labels)
    axs[2, 1].legend(loc='upper left', prop = {'size': legend_size})
    resnet.extend(nndt3)
    axs[2, 1].set_ylim([0, int(max(resnet)*1.4)])

    axs[2, 1].bar_label(rects1, padding=3)
    axs[2, 1].bar_label(rects2, padding=3)

    axs[2, 1].set_title('Targeted Attack 3\nSpanning NNDT4 Classifiers')

    plt.suptitle("1, 3, and 5-Pixel Attack Results")
    fig.tight_layout()
    plt.show()

def generate_one():
    labels = ["1", "3", "5"]
    rn = "ResNet18"
    n3 = "NNDT3"
    n4 = "NNDT4"
    resnet = [0, 2, 4]
    nndt3 = [0, 0, 1]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, resnet, width, label=rn)
    rects2 = ax.bar(x + width / 2, nndt3, width, label=n3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Successful Adversarial Images')
    ax.set_ylabel('Number of Pixels Attacked')
    ax.set_title('Untargeted Attack 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    resnet.extend(nndt3)
    ax.set_ylim([0, int(max(resnet)*1.4)])

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    #fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    generate_n_pixel_graphs()
    #generate_MIA_graphs()