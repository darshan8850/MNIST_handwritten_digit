import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

model = tf.keras.models.load_model("models/data/model")

app = Flask(__name__)

def preprocess_image(image_file):
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.resize((28, 28))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=3)
    return img_array

@app.route("/predict", methods=["GET"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = preprocess_image(image_file)
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)
        return jsonify({"predicted_label": int(predicted_label)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
