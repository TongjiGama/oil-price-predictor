import sys
import os

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import torch
import pandas as pd 
from pandas import DataFrame 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datapreprocessing.dataset import create_dataset

class OilDataset(Dataset):
    def __init__(self, args):
        X, y = create_dataset(args)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

    