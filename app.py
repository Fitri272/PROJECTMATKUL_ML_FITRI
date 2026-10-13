from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Memuat model Machine Learning
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "model_skincare.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    if request.method == "POST":
        ingredients = request.form["ingredients"]
        price = float(request.form["price"])

        data_baru = pd.DataFrame({
            "clean_ingreds": [ingredients],
            "price": [price]
        })

        hasil = model.predict(data_baru)[0]

    return render_template("index.html", hasil=hasil)


if __name__ == "__main__":
    app.run(debug=True)
