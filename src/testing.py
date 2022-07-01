import os

import torch
import torch.nn as nn
import numpy as np

import data_loader
import model
import utility

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

input_size = 10000
batch_size = 50
model_path = "src/models/model_epoch_ 2.pth"

print("Loading data...")
dataloaders = data_loader.get_loader(
    'yelp_ridotto3.json', batch_size)
test_loader = dataloaders['test']

print("==================================================================================")

print("Loading model...")
DAttn = model.DAttn(input_size)
DAttn.load_state_dict(torch.load(model_path))

if torch.cuda.is_available():
    DAttn.to(torch.device("cuda"))

criterion = nn.MSELoss()

print("==================================================================================")

print("Testing Start..")
test_loss = 0.0
DAttn.eval()
for i, (user, item, labels) in enumerate(test_loader):
    # Convert torch tensor to Variable
    user = utility.to_var(user)
    item = utility.to_var(item)
    labels = utility.to_var(labels.squeeze())
    outputs = DAttn(user, item)
    loss = criterion(outputs, labels)
    test_loss += loss.item()
print(f'Test Loss: {test_loss/len(test_loader)}')

print("==================================================================================")

print("Testing End..")
