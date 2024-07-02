import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import missingno as msno
from imblearn.combine import SMOTETomek

df = pd.read_csv("database/origin/healthcare-dataset-stroke-data.csv")

print(df.head())
print(df.isna().sum(), "Dados ausentes")
msno.bar(df)

imputer = SimpleImputer(strategy="mean")
df[["bmi"]] = imputer.fit_transform(df[["bmi"]])

print(df.isna().sum())

label_encoder = preprocessing.LabelEncoder()
df = df.apply(label_encoder.fit_transform)
print(df.head())

df["stroke"] = df["stroke"].replace({1: "YES", 0: "NO"})

X = df.drop("stroke", axis=1)
y = df["stroke"]

os = SMOTETomek(sampling_strategy=0.6)
X_res, y_res = os.fit_resample(X, y)

df_res = pd.concat(
    [
        pd.DataFrame(X_res, columns=X.columns),
        pd.DataFrame(y_res, columns=["stroke"]),
    ],
    axis=1,
)

df_res.to_csv("database/formated/encoded_df.csv", index=False)
print(df_res["stroke"].value_counts())
