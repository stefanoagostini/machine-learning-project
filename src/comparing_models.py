import data_loader

batch_size = 1


dataloaders = data_loader.get_loader(
    'yelp_ridotto3.json', batch_size)
train_loader = dataloaders['train']
val_loader = dataloaders['val']

test = train_loader.dataset.dataset.get_item_for_test()

print(test[0])  # stringa della recensione
print(test[1])  # user review
print(test[2])  # item review
print(test[3])  # rating/target
