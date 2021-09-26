import numpy as np
from numpy.random import default_rng
import pandas as pd
from torch.utils.data import Dataset
import torch

rng = default_rng()

class BikeDataset(Dataset):
    """Loads the Bike data from CSV into a torch Dataset."""
    train_prop = 0.8

    def __init__(self, root, train=True, **kwargs):
        df = pd.read_csv(root, parse_dates=['dteday'])
        df['dteday'] = df['dteday'].apply(lambda x: int(x.strftime('%d')))
        df.drop(columns=['instant', 'casual', 'registered'], inplace=True)
        self.data = df.values
        rng.shuffle(self.data)
        self.columns = df.columns
        n_train_data = int(self.train_prop * len(self.data))
        if train:
            self.data = self.data[:n_train_data]
        else:
            self.data = self.data[n_train_data:]
        self.data = torch.as_tensor(self.data).float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, -1]
