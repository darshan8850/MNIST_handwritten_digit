import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
from tensorflow.keras.datasets import mnist

# Data Load
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, x_test.shape)

#model
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(72, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(48, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(24, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

x_train = x_train.reshape(60000, 28, 28, 1)

x_test = x_test.reshape(10000, 28, 28, 1)

with mlflow.start_run() as run:
    
    # Log parameters
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 64)

    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)

    mlflow.tensorflow.save_model(model, "models")