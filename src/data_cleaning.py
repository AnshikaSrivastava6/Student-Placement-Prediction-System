import pandas as pd
from config import DataPath

df=pd.read_csv(DataPath)
print(df.isnull().sum())
df.drop("StudentID",axis=1,inplace=True)
# label encoding
col=["ExtracurricularActivities", "PlacementTraining"]
for i in col:
    df[i]=df[i].map({'Yes':1,'No':0})
df['PlacementStatus']=df['PlacementStatus'].map({'NotPlaced':0,'Placed':1})

print(df.head())
print("Total number of duplicates present in the dataset")
print(df.duplicated().sum())
pd.set_option("display.max_columns", None)

duplicate_rows = df[df.duplicated(keep=False)]

print(duplicate_rows.head(10))

df = df.drop_duplicates()
print(df.shape)
print(df.duplicated().sum())
df.to_csv("dataset/Cleaned_Dataset.csv",index=False)
print("cleaned data saved successfully")

print(df["PlacementStatus"].value_counts())
