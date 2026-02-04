from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    vec = vectorizer.transform([news])
    proba = model.predict_proba(vec)[0]

    fake_prob = proba[0]
    real_prob = proba[1]

    # Fake-biased threshold
    if fake_prob >= 0.55:
        label = f"FAKE NEWS ❌ (Confidence: {fake_prob*100:.2f}%)"
        color = "danger"
    else:
        label = f"REAL NEWS ✅ (Confidence: {real_prob*100:.2f}%)"
        color = "success"

    return render_template(
        "index.html",
        prediction=label,
        color=color
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
