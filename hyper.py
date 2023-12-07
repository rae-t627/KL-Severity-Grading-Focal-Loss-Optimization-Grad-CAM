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

batch_size = 32
frame_size = (224, 224)


import random
import custom_resnets
# import custom_densenets
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Set the parameter for reproducible results
random_seed = 21 # 21, 42 or 84

train_dataset_path = 'dataset/train'
val_dataset_path = 'dataset/val'
classes_header = ["0", "1", "2", "3", "4"] 
n_classes = len(classes_header)

# Choose CNN architecture and output directory

# cnn_model = custom_densenets.se_densenet121_model(n_classes)
# cnn_model = custom_densenets.densenet121_model(n_classes)
# cnn_model = custom_resnets.resnet18_model(n_classes)
# cnn_model = custom_resnets.resnet34_model(n_classes)
# cnn_model = custom_resnets.resnet50_model(n_classes)
cnn_model = custom_resnets.se_resnet18_model(n_classes)
# cnn_model = custom_resnets.se_resnet34_model(n_classes)
# cnn_model = custom_resnets.se_resnet50_model(n_classes)

# cnn_model = custom_resnets.se_resnet18_model(n_classes)
cnn_model.load_state_dict(torch.load('models/SE_ResNet_FL_64.25.ckpt', map_location=torch.device(device)))

checkpoints_dir = os.path.join('output', 'testing')

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

import sklearn

validation_split = .125

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
print(class_weights)

# Set up data loaders for training and validation
batch_size = 32  # You can adjust this based on your needs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)



if torch.cuda.is_available(): # Let's make sure GPU is available!
    device = torch.device("cuda:0")
    device_name = 'cuda:0'
    print("device: CUDA")
else:
    device = torch.device("cpu")
    device_name = 'cpu'
    print("device: CPU")


import re
import os
import utils


def append_to_file(filename, val):
    with open(filename, 'a') as f:
        f.write("%s\n" % val)

def read_file(filename):
    lines = []
    with open(filename, 'r') as f:
        lines = [float(line.strip()) for line in f]

    return lines

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, prev_val_acc = 0, lr_scheduler = None, anneal_epoch = 0, alpha = 2, gamma = 0.5): 
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

            ce_loss = loss(prediction, y_gpu)
            pt = torch.exp(-ce_loss)
            loss_value = torch.mean(alpha * (1 - pt) ** gamma * ce_loss)

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
    
    model_save_name = 'final.ckpt'
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, F"{model_save_name}"))
    print("final best accuracy: %f" % (best_val_accuracy))

    return loss_history, train_history, val_history
        

# train model
if device_name.startswith('cpu'):
    cnn_model.type(torch.FloatTensor)
    cnn_model.to(device)
else:
    cnn_model.type(torch.cuda.FloatTensor)
    cnn_model.to(device)

# loss = nn.CrossEntropyLoss(weight=class_weights).type(torch.cuda.FloatTensor)
# optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95) # decrease lr by 5% every 5 epochs
# prev_val_accuracy = 60

# loss_history, train_history, val_history = \
#     train_model(cnn_model, train_loader, val_loader, loss, optimizer, 71, prev_val_accuracy, lr_scheduler)

from optuna import Trial, create_study
from optuna.samplers import TPESampler

def objective(trial: Trial):
    # Hyperparameter search space
    alpha = trial.suggest_float('alpha', 0.5, 2.0)
    gamma = trial.suggest_float('gamma', 0.5, 2.0)
    
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    loss = nn.CrossEntropyLoss(weight=class_weights).type(torch.cuda.FloatTensor)
    
    checkpoints_dir = os.path.join('output', str(alpha), '_', str(gamma))

    # Modify your training function to accept alpha and gamma as arguments
    loss_history, train_history, val_history = train_model(
        cnn_model, train_loader, val_loader, loss, optimizer, 10,
        0, lr_scheduler, alpha=alpha, gamma=gamma
    )

    # Return the validation accuracy as the objective to maximize
    return max(val_history)

# Set up the hyperparameter optimization study
sampler = TPESampler(seed=random_seed)
study = create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=5)  # You can adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params
best_alpha = best_params['alpha']
best_gamma = best_params['gamma']

print("Best Alpha:", best_alpha)
print("Best Gamma:", best_gamma)
