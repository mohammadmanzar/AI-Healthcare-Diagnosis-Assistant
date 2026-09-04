# AI-Powered Healthcare Diagnosis Assistant

A simple academic machine-learning web application that accepts a natural-language symptom description and predicts a disease class using a TF-IDF + Logistic Regression model.

## Important limitation
This is an educational prototype. It is not a medical diagnostic device and must not be used as a substitute for a qualified healthcare professional.

## Dataset
The project uses the public Symptom2Disease dataset. It contains 1,200 natural-language symptom descriptions across 24 disease classes.

Source: https://www.kaggle.com/datasets/niyarrbarman/symptom2disease

A public CSV mirror is used by `download_dataset.py`: https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Download the dataset: `python download_dataset.py`
3. Train the model: `python train_model.py`
4. Start the application: `python app.py`
5. Open the local Flask address shown in the terminal.

Use only the metrics printed by `train_model.py` in the final report; do not invent results.
