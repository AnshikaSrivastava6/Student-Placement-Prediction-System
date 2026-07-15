import pandas as pd
from config import DataPath

df=pd.read_csv(DataPath)

print(df.head())

#explore dataset
print("Number of rows and columns present in the dataset")
print(df.shape)
print("Names of columns present in the dataset")
print(df.columns.to_list())
print("Information about the dataset")
print(df.info())
print("description of the dataset")
print(df.describe())

print(df['PlacementStatus'].value_counts())