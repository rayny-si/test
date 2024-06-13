import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from PIL import Image, ImageOps
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = load_model('spiral_keras_model.h5')
class_names = ['1 Healthy-Spiral', '0 Parkinsons-Spiral']  # Define your actual class names

def predict_image(model, img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return np.argmax(prediction)

# Assuming test data is stored as 'spiral/testing/healthy' and 'spiral/testing/parkinson'
test_dirs = {'spiral/testing/healthy': 1, 'spiral/testing/parkinson': 0}
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