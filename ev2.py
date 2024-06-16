import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D
from PIL import Image, ImageOps
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Custom BatchNormalization class to handle the axis parameter
class CustomBatchNormalization(BatchNormalization):
    def __init__(self, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        if isinstance(axis, list):
            axis = axis[0]
        super().__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = [self.axis]  # Ensure axis is saved as a list
        return config

# Custom DepthwiseConv2D class to handle the groups parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        if 'groups' in config:
            del config['groups']
        return config

# Load the model with custom objects
model = load_model('spiral_keras_model.h5', custom_objects={
    'BatchNormalization': CustomBatchNormalization,
    'DepthwiseConv2D': CustomDepthwiseConv2D
})

class_names = ['0 Healthy_Spiral', '1 Parkinson_Spiral']  # Define your actual class names

def predict_image(model, img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.LANCZOS)  # Corrected Resampling
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return np.argmax(prediction, axis=1)[0]  # Ensure correct dimension

# Assuming test data is stored as 'spiral/testing/healthy' and 'spiral/testing/parkinson'
test_dirs = {'spiral/testing/healthy': 0, 'spiral/testing/parkinson': 1}
y_true = []
y_pred = []
for dir_path, class_id in test_dirs.items():
    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file types if necessary
            img_path = os.path.join(dir_path, filename)
            pred_class_id = predict_image(model, img_path)
            y_true.append(class_id)
            y_pred.append(pred_class_id)

# Convert to numpy arrays for use with sklearn metrics
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')  # Use 'binary' if you have only two classes
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)