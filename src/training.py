import os

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

import data_loader
import model
import utility

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

input_size = 10000
num_epochs = 10
batch_size = 50
learning_rate = 1e-4

print("Loading data...")
dataloaders = data_loader.get_loader(
    'yelp_ridotto3.json', batch_size)
train_loader = dataloaders['train']
val_loader = dataloaders['val']
print("==================================================================================")

DAttn = model.DAttn(input_size)

if torch.cuda.is_available():
    print("GPU enabled")
    DAttn.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(DAttn.parameters(), lr=learning_rate)

print("==================================================================================")
print("Training Start..")

min_valid_loss = np.inf

# Train the Model
total_step_train = len(train_loader) - 1
total_step_val = len(val_loader) - 1
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    DAttn.train()
    for i, (user, item, labels) in enumerate(train_loader):
        user = utility.to_var(user)
        item = utility.to_var(item)
        labels = utility.to_var(labels.squeeze())
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = DAttn(user, item)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if(i % 100 == 0):
            print(
                f'Epoch: {epoch+1} \t\t Training Percent: {i/total_step_train * 100}%')

    print("Validating")
    DAttn.eval()
    for i, (user, item, labels) in enumerate(val_loader):
        user = utility.to_var(user)
        item = utility.to_var(item)
        labels = utility.to_var(labels.squeeze())
        outputs = DAttn(user, item)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        if(i % 10 == 0):
            print(
                f'Epoch: {epoch+1} \t\t Validation Percent: {i/total_step_val * 100}%')

    # Print log info
    print(
        f'Epoch: {epoch+1}/{num_epochs} \t\t Training Loss: {train_loss / len(train_loader)}  \t\t Validation Loss: {val_loss / len(val_loader)}')

    if min_valid_loss > val_loss:
        print(
            f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        min_valid_loss = val_loss
        # Saving State Dict
        torch.save(DAttn.state_dict(),
                   f'src/models/model_epoch_ {str(epoch)}.pth')

print("==================================================================================")
print("Training End..")
