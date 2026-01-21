import sys
sys.path.insert(0, './')

import pandas as pd 
from pandas import DataFrame 
import numpy as np
from torch.utils.data import Dataset, DataLoader


from dataProcessor.utils import create_data_set, get_dataloader

class EuropeBrentSpotPriceFOB(Dataset):
    def __init__(self, path):
        self.data:DataFrame = pd.read_csv(path)
        self.X, self.y = create_data_set(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #  batch must contain tensors, numpy arrays, numbers, dicts or lists;
        X = self.X[idx]
        y = self.y[idx]
        # print(type(X), type(y))
        return X, y



if __name__ == "__main__":
    ds = EuropeBrentSpotPriceFOB("./data/RBRTEd.csv")
    dl = get_dataloader(ds)
    for batch in dl:
        print(batch.shape)
        break
    