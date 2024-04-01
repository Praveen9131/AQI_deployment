import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the saved model
model = load_model("alz_model1.h5")

# Function to preprocess the image and make a prediction
def predict_image(img):
    # Ensure the image is a PIL Image for resizing
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img.astype('uint8'), 'RGB')

    # Resize the image to match the model's expected input size
    img = img.resize((224, 224))

    # Convert the PIL image to a NumPy array
    img_array = np.array(img)

    # Scale the image data to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Class names, based on the model's training
    class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
    
    # Find the class with the highest probability
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Create a Gradio interface
iface = gr.Interface(fn=predict_image,
                     inputs=gr.Image(),
                     outputs='text',
                     title="Alzheimer's Detection Model Using 3D CNN by Sobitha",
                     description="Upload an image for Alzheimer's disease classification.")

# Launch the interface
iface.launch()
