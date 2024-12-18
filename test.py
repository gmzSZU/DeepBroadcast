import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import os
import random
import deepbroadcast
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
val_batch_size = 100

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

valset = torchvision.datasets.CIFAR10(root='./dataset/', train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=8)

model = deepbroadcast.DeepBroadcast()

ckp_path = "./results/cifar10/DeepBroadcast_best_model.pth"

msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'))
print(msg)
model.to(device)
model.eval()

snr_list = [-5, -2, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]
channel_flag = {'Rx1': 'Rayleigh', 'Rx2': 'AWGN'}
user_list = ['Rx1', 'Rx2']
val_snr_dict = {'Rx1': 20, 'Rx2': 20}

rx1_acc = []
rx2_acc = []
avg_acc = []

for i in range(len(snr_list)):
    val_snr_dict['Rx1'] = snr_list[i]
    for j in range(len(snr_list)):
        val_snr_dict['Rx2'] = snr_list[j]
        counter = 0
        acc1_total = 0
        acc2_total = 0
        val_acc = 0
        print ("In this test, the channel condition is " + str(val_snr_dict))
        for imgs, labels in tqdm(valloader):
            counter += 1
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            res1, res2, _ = model(imgs, val_snr_dict, channel_flag)
            new_labels1 = utils.cifar10_label_trans(labels, labels.size(0), rx=1).to(device)
            new_labels2 = utils.cifar10_label_trans(labels, labels.size(0), rx=2).to(device)
            
            
            _, predicted1 = torch.max(res1.data, 1)
            _, gt1 = torch.max(new_labels1.data, 1)
            count1 = predicted1.eq(gt1).cpu().sum()
            acc1 = predicted1.eq(gt1).cpu().sum() / new_labels1.size(0)
            acc1_total += acc1
            
            _, predicted2 = torch.max(res2.data, 1)
            _, gt2 = torch.max(new_labels2.data, 1)
            count2 = predicted2.eq(gt2).cpu().sum()
            acc2 = predicted2.eq(gt2).cpu().sum() / new_labels2.size(0)
            acc2_total += acc2
            
            val_acc += (acc1+acc2) / 2    
        
        acc1_avg = 100 * acc1_total / counter
        rx1_acc.append(acc1_avg)
        acc2_avg = 100 * acc2_total / counter
        rx2_acc.append(acc2_avg)
        avg_acc_local = 100*val_acc/counter
        avg_acc.append(avg_acc_local)
        print ("Rx1 Acc: %.3f || Rx2 Acc: %.3f" % (acc1_avg, acc2_avg))
        print("Avg Val Accuracy: %.03f%%" % (100*val_acc/counter))
        
np.save("./rx1_acc.npy", np.array(rx1_acc))
np.save("./rx2_acc.npy", np.array(rx2_acc))
np.save("./avg_acc.npy", np.array(avg_acc))
