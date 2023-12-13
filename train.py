from torchvision import transforms
from torchvision import datasets
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torchvision import datasets

import random
import custom_resnets
import custom_densenets
import os

import sklearn

import re
import utils
import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-m', '--model', type=str, choices = ["densenet", "resnet"], default="resnet")
parser.add_argument('-w', '--weights', type = str, help = "path to weights", default=None)
parser.add_argument('-p', '--pretrained', action="store_true", help="train on top of pretrained model")
parser.add_argument('-d', '--dataset', type = str, help = "path to dataset", default='dataset')
parser.add_argument('-b', '--batch_size', type=int, choices=[16, 32, 64, 128, 256], default=32)
parser.add_argument('-l', '--loss', type=str, help="loss function to be used", choices=["fl", "ce"], default="ce")
parser.add_argument('-o', '--output_dir', type=str, default="output")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)

args = parser.parse_args()

frame_size = (224, 224)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Set the parameter for reproducible results
random_seed = 21 # 21, 42 or 84

train_dataset_path = os.path.join(args.dataset, "train")
val_dataset_path = os.path.join(args.dataset, "val")
classes_header = ["0", "1", "2", "3", "4"] 
n_classes = len(classes_header)

# choose an architecture
if (args.model == "densenet"):
    cnn_model = custom_densenets.se_densenet121_model(n_classes)
else:
    cnn_model = custom_resnets.se_resnet18_model(n_classes)

if (args.pretrained):
    cnn_model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))

checkpoints_dir = args.output_dir

if (not os.path.exists(checkpoints_dir)):
    os.mkdir(checkpoints_dir)


random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# if you are using GPU
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


#Loading the dataset
transforms_to_train = transforms.Compose([         
    transforms.ColorJitter(brightness=.33, saturation=.33),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.10)),
    transforms.Resize(frame_size), 

    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(train_dataset_path, transform=transforms_to_train)
val_dataset = datasets.ImageFolder(val_dataset_path, transform=transforms_to_train)

# Access the targets (labels) for training and validation
train_targets = train_dataset.targets
val_targets = val_dataset.targets

# Set up samplers for training and validation
train_sampler = SubsetRandomSampler(range(len(train_dataset)))
val_sampler = SubsetRandomSampler(range(len(val_dataset)))

class_weights=sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
class_weights = torch.FloatTensor(class_weights)

# Set up data loaders for training and validation
batch_size = args.batch_size  # You can adjust this based on your needs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)

# Making sure GPU is available
if torch.cuda.is_available(): 
    device = torch.device("cuda:0")
    device_name = 'cuda:0'
    print("device: CUDA")
else:
    device = torch.device("cpu")
    device_name = 'cpu'
    print("device: CPU")


def append_to_file(filename, val):
    with open(filename, 'a') as f:
        f.write("%s\n" % val)

def read_file(filename):
    lines = []
    with open(filename, 'r') as f:
        lines = [float(line.strip()) for line in f]

    return lines

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, prev_val_acc = 0, lr_scheduler = None, anneal_epoch = 0, alpha = None, gamma = 2): 
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    start_epoch = 0

    open(os.path.join(checkpoints_dir,'loss_history.txt'), 'a').close()
    open(os.path.join(checkpoints_dir, 'train_history.txt'), 'a').close()
    open(os.path.join(checkpoints_dir, 'val_history.txt'), 'a').close()
    open(os.path.join(checkpoints_dir, 'best_accuracy.txt'), 'a').close()
    print("start training from scratch...")

    best_val_accuracy = prev_val_acc

    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        model.train() # Enter train mode
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0

        # process batches
        for i_step, (x, y) in enumerate(train_loader): 
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)   

            if (args.loss == "ce"):
                loss_value = loss(prediction, y_gpu)
            else:
                ce_loss = loss(prediction, y_gpu)
                pt = torch.exp(-ce_loss)
                loss_value = torch.mean(alpha[y_gpu] * (1 - pt) ** gamma * ce_loss)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            
            loss_accum += loss_value

        # check accuracy
        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) * 100 / total_samples

        val_accuracy = 0.0
        with torch.no_grad():
          val_accuracy, _, _, _ = utils.compute_accuracy(model, val_loader)

        val_history.append(val_accuracy)

        # write marks to files
        append_to_file(os.path.join(checkpoints_dir, 'loss_history.txt'), float(ave_loss))
        append_to_file(os.path.join(checkpoints_dir, 'train_history.txt'), train_accuracy)
        append_to_file(os.path.join(checkpoints_dir, 'val_history.txt'), val_accuracy)

        # update learning rate
        if lr_scheduler is not None and epoch >= anneal_epoch:
          lr_scheduler.step()
        
        stage = epoch + start_epoch

        if val_accuracy > best_val_accuracy:
          best_val_accuracy = val_accuracy
          
          # best model
          model_save_name = 'best_model.ckpt'
          torch.save(model.state_dict(), os.path.join(checkpoints_dir, F"{model_save_name}"))

          append_to_file(os.path.join(checkpoints_dir, 'best_accuracy.txt'), best_val_accuracy)
          print("update best model with val. accuracy %f on stage %d" % (best_val_accuracy, stage))

        print("epoch %d; average loss: %f, train accuracy: %f, val accuracy: %f" % (stage, ave_loss, train_accuracy, val_accuracy))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoints_dir, "final.pth"))

    print("final best accuracy: %f" % (best_val_accuracy))

    return loss_history, train_history, val_history
        

# train model
if device_name.startswith('cpu'):
    cnn_model.type(torch.FloatTensor)
    cnn_model.to(device)
else:
    cnn_model.type(torch.cuda.FloatTensor)
    cnn_model.to(device)

loss = nn.CrossEntropyLoss(weight=class_weights).type(torch.cuda.FloatTensor)
optimizer = optim.Adam(cnn_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95) # decrease lr by 5% every 5 epochs
prev_val_accuracy = 0

loss_history, train_history, val_history = \
    train_model(cnn_model, train_loader, val_loader, loss, optimizer, args.epochs, prev_val_accuracy, lr_scheduler, alpha = class_weights.to(device), gamma = 2.00)
