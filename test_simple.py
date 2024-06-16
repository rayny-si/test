import streamlit as st 
from PIL import Image 
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D

# # Initialize or load session state variables to manage the workflow
image_spiral= None
image_wave = None

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

def getmodel(modelname):
    model= load_model(modelname,  custom_objects={
        'BatchNormalization': CustomBatchNormalization,
        'DepthwiseConv2D': CustomDepthwiseConv2D
    })
    return model

# Load the models and class names spiral_model = get_spiral_model()
spiral_model =  getmodel("spiral_keras_model.h5")
spiral_class_names =  ['0 Healthy_Spiral', '1 Parkinson_spiral'] 

wave_model =  getmodel("wave_keras_model.h5")
wave_class_names =['0 Healthy_Wave', '1 Parkinson_Wave'] 

clock_model =  getmodel("clock_keras_model.h5")
clock_class_names =['0 Healthy_Clock', '1 Parkinson_Clock'] 

# Function to calculate final prediction
# sample value ('0_UnHealthy_Spiral',0.8,'1_Healthy_wave',0.4)
def calculate_final_prediction(class_spiral, confidence_spiral, class_wave, confidence_wave, class_clock, confidence_clock):
    adjusted_confidence_spiral = confidence_spiral if class_spiral == "0 Healthy_Spiral" else -confidence_spiral 
    adjusted_confidence_wave = confidence_wave if class_wave == "0 Healthy_Wave" else -confidence_wave
    adjusted_confidence_clock = confidence_clock if class_clock == "0 Healthy_Clock" else -confidence_clock  
    
    total_confidence = adjusted_confidence_spiral + adjusted_confidence_wave + adjusted_confidence_clock
    
    final_prediction = "This person is healthy." if total_confidence > 0 else "This person seems to have Parkinson's"
    
    return final_prediction, total_confidence

# Helper function to upload and display images
def upload_image(key):
    global image_spiral,image_wave,image_clock
    
    uploaded_file = st.file_uploader(f"Upload image for {key} model", key=key) 
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        if key == "spiral":
            image_spiral =image
        elif key == "wave":
            image_wave = image
        elif key == "clock":
            image_clock = image
        st.image(image) # display image

# Streamlit interface
st.title("Test Drawings for Parkinson's")

st.write("Enter wave, spiral and clock drawings to test for the disease.")

#Layout for model inputs
col_wave, col_spiral, col_clock=st.columns(3)
with col_wave:
    upload_image("wave")
with col_spiral:
    upload_image("spiral")
with col_clock:
    upload_image("clock")

def data_preprocessing(img):
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
 
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
 
    # Load the image into the array
    
    return normalized_image_array

def predict_image(model, class_names,image):

 # Predicts the model
    normalized_image_array = data_preprocessing(image)
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    # prediction = np.array([[0.2, 0.8]]) # sample
    
    index = np.argmax(prediction) 
    
    #  ['0_UnHealthy_Spiral', '1_Healthy_Spiral']  sample 
    predicted_classname = class_names[index] 
    confidence_score = prediction[0][index]
    return predicted_classname,confidence_score

# Ensure the clock model and class names are defined
# clock_model = ... (your clock model)
# clock_class_names = ... (your clock class names)

# Button to trigger predictions
if st.button("Test Results"):
    if image_spiral and image_wave and image_clock:

        class_name_wave, confidence_score_wave = predict_image(wave_model, wave_class_names, image_wave)
        class_name_spiral, confidence_score_spiral = predict_image(spiral_model, spiral_class_names, image_spiral)
        class_name_clock, confidence_score_clock = predict_image(clock_model, clock_class_names, image_clock)

        # Display individual predictions
        st.write(f"Wave Model Class: {class_name_wave}, Confidence Score: {confidence_score_wave:.2f}")
        st.write(f"Spiral Model Class: {class_name_spiral}, Confidence Score: {confidence_score_spiral:.2f}")
        st.write(f"Clock Model Class: {class_name_clock}, Confidence Score: {confidence_score_clock:.2f}")

        # Final Combined Prediction
        st.subheader("Final Prediction")
        final_prediction, total_confidence = calculate_final_prediction(class_name_spiral, confidence_score_spiral, class_name_wave, confidence_score_wave, class_name_clock, confidence_score_clock)
        st.success(f"Final Prediction: {final_prediction}")
        st.info(f"Total Confidence Value: {total_confidence:.2f}")
