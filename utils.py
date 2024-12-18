import torch 
import torch.nn as nn
import numpy as np
import random

def cifar10_label_trans(labels, batch_size, rx):
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new_labels = torch.zeros([batch_size, 2]).to(device)
    if rx == 1:
        for i in range(len(labels)):
            if labels[i] in [2, 3, 4, 5, 6, 7]:
                new_labels[i][0] = 1
            else:
                new_labels[i][1] = 1
    elif rx == 2:
        for i in range(len(labels)):
            if labels[i] in [1, 3, 4, 5, 7]:
                new_labels[i][0] = 1
            else:
                new_labels[i][1] = 1
    return new_labels


