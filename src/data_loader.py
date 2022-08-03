import pickle

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


r_dtypes = {'review_id': str,
            'user_id': str,
            'stars': int,
            'text': str}


class ReviewDataset(data.Dataset):

    def __init__(self, path, root='src/data/', max_len=10000):
        self.dataset = pd.read_json(
            root + path, dtype=r_dtypes)
        # load GloVe
        glove_path = root + 'glove.6B.300d.pickle'
        with open(glove_path, 'rb') as f:
            self.glove = pickle.load(f)
        np.random.seed(42)
        self.unknown = np.random.uniform(0, 1, 100)
        self.delimiter = np.random.uniform(0, 1, 100)
        self.max_len = max_len

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        user_id = row['user_id']  # serve per la query sul dataframe
        business_id = row['business_id']  # serve per la query sul dataframe
        # user review
        user_data = self.dataset.query(
            "user_id == @user_id").drop(columns=['user_id', 'business_id', 'stars'])
        user_review = self.preprocess_review(user_data)
        # item review
        business_data = self.dataset.query(
            "business_id == @business_id").drop(columns=['user_id', 'business_id', 'stars'])
        item_review = self.preprocess_review(business_data)
        # rating
        target = torch.Tensor([row['stars']]).float()
        if user_review is not None:
            user_review = torch.from_numpy(user_review).float()
        else:
            return None
        if item_review is not None:
            item_review = torch.from_numpy(item_review).float()
        else:
            return None
        return (user_review, item_review, target)

    def get_item_for_test(self):
        sample = self.dataset.sample(
            random_state=np.random.Generator(np.random.PCG64()))
        # user review
        user_review = self.preprocess_review(sample)
        # item review
        item_review = self.preprocess_review(sample)
        # rating
        target = torch.Tensor([sample['stars'].to_numpy()]).float()
        if user_review is not None:
            user_review = torch.from_numpy(user_review).float()
        else:
            return None
        if item_review is not None:
            item_review = torch.from_numpy(item_review).float()
        else:
            return None
        return [sample["text"], user_review, item_review, target]

    def __len__(self):
        """ Returns the number of reviews in the dataset"""
        return len(self.dataset)

    def preprocess_review(self, reviews: pd.DataFrame):
        reviews_len = len(reviews)
        if reviews_len > 100:
            reviews = reviews[:100]
        total_review = np.array([])
        for review_str in reviews['text']:
            review: list[str] = word_tokenize(review_str)
            total_review = np.concatenate((total_review, review))
            total_review = np.append(total_review, '+++')
        review = []
        for word in total_review:
            if word == '+++':
                review.append(self.delimiter)
            else:
                if word in self.glove:
                    review.append(self.glove[word])
                else:
                    review.append(self.unknown)
        review = np.array(review)
        if len(review) < self.max_len:
            pad_len = self.max_len - len(review)
            pad_vector = np.zeros((pad_len, 100))
            review = np.concatenate((review, pad_vector), axis=0)
        else:
            review = review[:self.max_len]
        return review


def train_val_dataset(dataset, val_split=0.1, test_split=0.1):
    """Split dataset in train, validation and test set"""
    train_idx, val_test_idx = train_test_split(
        list(range(len(dataset))), test_size=(val_split+test_split), random_state=42)
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=test_split, random_state=42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets


def get_loader(data_path, batch_size=100, shuffle=True, num_workers=2, max_len=10000):
    """Builds and returns Dataloader"""
    dataset = ReviewDataset(data_path, max_len=max_len)
    datasets = train_val_dataset(dataset)
    dataloaders = {x: data.DataLoader(
        dataset=datasets[x], batch_size=batch_size,
        shuffle=shuffle) for x in ['train', 'val', 'test']}
    return dataloaders
