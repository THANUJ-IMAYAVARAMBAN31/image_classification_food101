from flask import Flask, render_template, request, jsonify
from predict import predict_image

import requests
import os


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ── File upload path ──────────────────────────────────
        if "image" in request.files:

            file = request.files["image"]

            if file.filename == "":
                return jsonify({"error": "No file selected"})

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = predict_image(filepath)
            return jsonify(result)

        # ── URL path ──────────────────────────────────────────
        elif "url" in request.form:

            url = request.form["url"].strip()

            if not url:
                return jsonify({"error": "URL is empty"})

            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            filepath = os.path.join(UPLOAD_FOLDER, "url_image.jpg")

            with open(filepath, "wb") as f:
                f.write(response.content)

            result = predict_image(filepath)
            return jsonify(result)

        # ── Neither field sent ────────────────────────────────
        else:
            return jsonify({"error": "No image file or URL provided"})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)