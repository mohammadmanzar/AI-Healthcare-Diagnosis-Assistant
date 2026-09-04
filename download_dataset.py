import pandas as pd
URL = "https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv"
df = pd.read_csv(URL, index_col=0)
df.to_csv("dataset.csv", index=False)
print("Dataset downloaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
