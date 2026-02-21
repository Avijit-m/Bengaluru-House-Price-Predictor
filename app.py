from flask import Flask, render_template, request
import json
import pickle
import numpy as np
import os


app = Flask(__name__)


# Load model and columns metadata once at startup
MODEL_FILENAME = "banglore_home_prices_model.pickle"
COLUMNS_FILENAME = "columns.json"

model = None
data_columns = None
location_columns = None


def load_artifacts():
    global model, data_columns, location_columns

    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, COLUMNS_FILENAME), "r") as f:
        columns = json.load(f)
        data_columns = columns["data_columns"]

    # first three columns are numeric [total_sqft, bath, bhk], rest are locations
    location_columns = data_columns[3:]

    with open(os.path.join(base_dir, MODEL_FILENAME), "rb") as f:
        model = pickle.load(f)


def get_estimated_price(location, sqft, bhk, bath):
    """Build feature vector in same order as training and return predicted price."""
    x = np.zeros(len(data_columns))

    # numeric features
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # one-hot location (all columns in JSON are already lower-cased)
    loc = location.strip().lower()
    if loc in data_columns:
        loc_index = data_columns.index(loc)
        x[loc_index] = 1

    return float(model.predict([x])[0])


@app.route("/", methods=["GET", "POST"])
def index():
    if model is None or data_columns is None:
        load_artifacts()

    locations = sorted(location_columns)
    predicted_price = None

    if request.method == "POST":
        try:
            location = request.form.get("location") or ""
            sqft = float(request.form.get("sqft") or 0)
            bhk = int(request.form.get("bhk") or 0)
            bath = int(request.form.get("bath") or 0)

            predicted_price = round(get_estimated_price(location, sqft, bhk, bath), 2)
        except Exception:
            predicted_price = None

    return render_template(
        "index.html",
        locations=locations,
        predicted_price=predicted_price,
    )


@app.route("/notebook")
def notebook():
    """
    Simple route to showcase the notebook.
    Assumes you will export the notebook to HTML as `notebook.html` in the same directory.
    """
    return render_template("notebook.html")


if __name__ == "__main__":
    # Enable debug for easier local development; you can turn this off in production.
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)

