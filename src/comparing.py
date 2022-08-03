import os

import torch
import torch.nn as nn

import data_loader
import model
import utility

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

input_size = 10000
batch_size = 50
model_path = "src/models/model_epoch_ 2.pth"
model_path_lattn_only = "src/models/model_epoch_lattn_ 2.pth"
model_path_gattn_only = "src/models/model_epoch_GAttn_ 5.pth"

print("Loading data...")
dataloaders = data_loader.get_loader(
    'yelp_ridotto3.json', batch_size)
test_loader = dataloaders['test']


print("==================================================================================")

print("Sample input...")

input = test_loader.dataset.dataset.get_item_for_test()

print(input[0])

print("==================================================================================")


print("Loading model...")
DAttn = model.DAttn(input_size)
GAttOnly = model.GAttOnly(input_size)
LAttOnly = model.LAttOnly(input_size)


DAttn.load_state_dict(torch.load(model_path))
GAttOnly.load_state_dict(torch.load(model_path_gattn_only))
LAttOnly.load_state_dict(torch.load(model_path_lattn_only))

if torch.cuda.is_available():
    DAttn.to(torch.device("cuda"))
    GAttOnly.to(torch.device("cuda"))
    LAttOnly.to(torch.device("cuda"))


criterion = nn.MSELoss()

print("==================================================================================")

print("Comparing...")

DAttn.eval()
GAttOnly.eval()
LAttOnly.eval()


user = utility.to_var(input[1])
item = utility.to_var(input[2])
labels = utility.to_var(input[3].squeeze())

print("DAttn")
outputs_DAttn = DAttn(user, item)
loss = criterion(outputs_DAttn, labels)
test_loss = loss.item()
print(f'Test Loss: {test_loss}')

print("GAttOnly")
outputs_GAttOnly = GAttOnly(user, item)
loss = criterion(outputs_GAttOnly, labels)
test_loss = loss.item()
print(f'Test Loss: {test_loss}')

print("LAttOnly")
outputs_LAttOnly = LAttOnly(user, item)
loss = criterion(outputs_LAttOnly, labels)
test_loss = loss.item()
print(f'Test Loss: {test_loss}')
