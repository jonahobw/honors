from torchvision import datasets, transforms
import os
import torch
import random
import datetime
import time

a = os.listdir(os.path.join("small_test_dataset", "Test"))
for i,x in enumerate(a):
    print('Test/{}, i = {}'.format(x, i))