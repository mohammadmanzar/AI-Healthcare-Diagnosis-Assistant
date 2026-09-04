import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATASET_URL = "https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv"
DATASET_FILE = "dataset.csv"

try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print("dataset.csv not found. Downloading from the public dataset mirror...")
    df = pd.read_csv(DATASET_URL, index_col=0)
    df.to_csv(DATASET_FILE, index=False)

df = df[["label", "text"]].dropna().drop_duplicates()
df["label"] = df["label"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

print("Dataset shape:", df.shape)
print("Number of disease classes:", df["label"].nunique())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
