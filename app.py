from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    model = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check if model exists
    if model is None:
        return render_template(
            "result.html",
            error="Model not found. Please run train_model.py first."
        )

    # Get symptoms from user
    symptoms = request.form.get("symptoms", "").strip()

    if not symptoms:
        return render_template(
            "result.html",
            error="Please enter at least one symptom."
        )

    # Main prediction
    prediction = model.predict([symptoms])[0]

    # Get probabilities
    top_predictions = []

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba([symptoms])[0]
        classes = model.classes_

        # Get indexes of top 3 probabilities
        top_indices = probabilities.argsort()[-3:][::-1]

        for index in top_indices:
            top_predictions.append({
                "disease": classes[index],
                "probability": round(float(probabilities[index]) * 100, 2)
            })

    # Main confidence
    confidence = None

    if top_predictions:
        confidence = top_predictions[0]["probability"]

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        symptoms=symptoms,
        top_predictions=top_predictions
    )


if __name__ == "__main__":
    app.run(debug=True)