import data_loader

batch_size = 1


dataloaders = data_loader.get_loader(
    'yelp_ridotto3.json', batch_size)
train_loader = dataloaders['train']
val_loader = dataloaders['val']

print(train_loader.dataset.dataset.get_item_for_test())
