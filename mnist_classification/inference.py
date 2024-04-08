import tensorflow as tf
from flask import Flask, request, jsonify

model = tf.keras.models.load_model("models")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['image']
    data = tf.convert_to_tensor(data)
    data = tf.expand_dims(data, axis=0)
    
    predictions = model.predict(data)
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
    
    return jsonify({"predicted_label": int(predicted_label)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
