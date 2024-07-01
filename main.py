import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import missingno as msno

df = pd.read_csv("database/origin/healthcare-dataset-stroke-data.csv")

print(df.head())
print(df.isna().sum(), "Dados ausentes")
print(msno.bar(df))

imputer = SimpleImputer(strategy="mean")
imputer.fit_transform
df[["bmi"]] = imputer.fit_transform(df[["bmi"]])

print(df.isna().sum())

label_encoder = preprocessing.LabelEncoder()
df = df.apply(label_encoder.fit_transform)
print(df.head())

df["stroke"] = df["stroke"].replace({1: "YES", 0: "NO"})

df.to_csv("database/formated/encoded_df.csv", index=False)
