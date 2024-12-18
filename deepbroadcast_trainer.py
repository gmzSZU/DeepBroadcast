# =============================================================================
# THIS repo is contributed by Mingze Gong (Graduated Student Member, IEEE) and 
# Shuoyao Wang (Senior Member, IEEE), Shenzhen University, Shenzhen, China. 
# This repo aims at introducing the DeepBroadcast method, with the 2-user case 
# study. The following is the training part. 
# If you found this project helpful to your research, please cite our PAPER:
# @ARTICLE{10738311,
#   author={Gong, Mingze and Wang, Shuoyao and Ye, Fangwei and Bi, Suzhi},
#   journal={IEEE Transactions on Wireless Communications}, 
#   title={Compression Before Fusion: Broadcast Semantic Communication System 
#   for Heterogeneous Tasks}, 
#   year={2024},
#   volume={23},
#   number={12},
#   pages={19428-19443},
#   keywords={Receivers;Feature extraction;Wireless communication;Image 
#   reconstruction;Transmitters;Data mining;Transform coding;Wireless sensor 
#   networks;Training;Simulation;Semantic communication;broadcast;multi-task; 
#   heterogeneous channels},
#   doi={10.1109/TWC.2024.3483314}}
# =============================================================================
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
total_epoch = 200
best_acc = 80
train_batch_size = 256
val_batch_size = 100
save_path = "./results/cifar10/"

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


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
valset = torchvision.datasets.CIFAR10(root='./dataset/', train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=8)

model = deepbroadcast.DeepBroadcast().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
optimizer.zero_grad()
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-5)

# wirless transmission simulation settings 
snr_list = [-5, -2, 1, 4, 7, 10]
channel_flag = {'Rx1': 'Rayleigh', 'Rx2': 'AWGN'}
user_list = ['Rx1', 'Rx2']
val_snr_dict = {'Rx1': 11, 'Rx2': 11}

for epoch in range(total_epoch):
    model.train()
    total_loss_epoch = 0
    acc_epoch = 0
    num_iters = len(trainloader)
    print ("\n")
    print ("<<<Epoch %d>>>" % (int(epoch)))
    print ("Current Learning Rate: " + str(optimizer.state_dict()['param_groups'][0]['lr']))
    for imgs, labels in tqdm(iter(trainloader)):
        train_snr_dict = utils.random_snr(snr_list, user_list)
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        res1, res2, KL_loss = model(imgs, train_snr_dict, channel_flag)
        new_labels1 = utils.cifar10_label_trans(labels, labels.size(0), rx=1).to(device)
        new_labels2 = utils.cifar10_label_trans(labels, labels.size(0), rx=2).to(device)
        
        loss1 = loss_fn(res1, new_labels1)
        loss2 = loss_fn(res2, new_labels2)
        loss = (loss1+loss2)/2 + KL_loss*1e-4
        loss.backward()
        optimizer.step()
        
        total_loss_epoch += loss.item()
        _, predicted1 = torch.max(res1.data, 1)
        _, gt1 = torch.max(new_labels1.data, 1)
        acc1 = predicted1.eq(gt1).cpu().sum() / new_labels1.size(0)
        _, predicted2 = torch.max(res2.data, 1)
        _, gt2 = torch.max(new_labels2.data, 1)
        acc2 = predicted2.eq(gt2).cpu().sum() / new_labels2.size(0)
        
        acc_epoch += (acc1+acc2)/2
    scheduler.step()
    print("Epoch: %d || Avg Train Loss: %.05f || Avg Train Accuracy: %.03f%% " % (epoch, total_loss_epoch/num_iters, 100*acc_epoch/num_iters))
    
    if epoch >= 100:
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            counter = 0
            model.eval()
            for imgs, labels in tqdm(valloader):
                counter += 1
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                res1, res2 = model(imgs, val_snr_dict, channel_flag)
                new_labels1 = utils.cifar10_label_trans(labels, labels.size(0), rx=1).to(device)
                new_labels2 = utils.cifar10_label_trans(labels, labels.size(0), rx=2).to(device)
                
                loss1 = loss_fn(res1, new_labels1)
                loss2 = loss_fn(res2, new_labels2)
                loss = (loss1 + loss2) / 2

                val_loss += loss.item()
                
                _, predicted1 = torch.max(res1.data, 1)
                _, gt1 = torch.max(new_labels1.data, 1)
                acc1 = predicted1.eq(gt1).cpu().sum() / new_labels1.size(0)
                _, predicted2 = torch.max(res2.data, 1)
                _, gt2 = torch.max(new_labels2.data, 1)
                acc2 = predicted2.eq(gt2).cpu().sum() / new_labels2.size(0)
                val_acc = (acc1+acc2) / 2 
                
            print("Val Result: Avg Val Loss: %.05f || Avg Val Accuracy: %.03f%%" % (val_loss/counter, 100*val_acc/counter))
            
            if 100*val_acc/counter > best_acc:
                best_acc = 100*val_acc/counter
                best_epoch = epoch
                print ("New Record Confirm, Saving Model...")
                torch.save(model.state_dict(), save_path+"DeepBroadcast_best_model.pth")
                
print("All training process for DeepBroadcast has been done.")
print ("Best Avg Acc: %.05f" % (best_acc))
print ("Best Epoch: %d" % (best_epoch))
