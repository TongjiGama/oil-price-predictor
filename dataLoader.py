import pandas as pd 
from pandas import DataFrame as df




if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    print(data.head())