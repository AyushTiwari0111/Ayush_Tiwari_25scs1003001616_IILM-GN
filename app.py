from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("mnist_cnn.h5")

@app.route("/predict", methods=["POST"])
def predict():
    # Get uploaded file
    file = request.files["file"]

    # Load and convert to grayscale
    img = Image.open(file).convert("L")

    # Resize to 28x28 to match MNIST
    img = img.resize((28, 28))

    # Convert to numpy array
    img = np.array(img).astype("float32") / 255.0

    # MNIST has white digits on black background
    # If the uploaded image is mostly white, invert it
    if img.mean() > 0.5:
        img = 1 - img

    # Reshape for the model
    img = img.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img)
    predicted_digit = int(np.argmax(prediction))

    # Return result
    return jsonify({"prediction": predicted_digit})


if __name__ == "__main__":
    app.run(debug=True)
