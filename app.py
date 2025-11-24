import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from gradcam import generate_gradcam

IMG_SIZE = 150

# Load model
model = load_model("model.h5")
model.build((None, 150, 150, 3))   # 👈 ADD THIS
model(img_array_example := np.zeros((1,150,150,3)))  # 👈 Force model to build


# IMPORTANT FIX for "sequential has no defined output"
model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
model.call(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)))
print("MODEL BUILT SUCCESSFULLY!")

def preprocess(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0)

def predict(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = preprocess(img)

    pred = model.predict(x)[0][0]

    heatmap = generate_gradcam(model, x)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    label = "Tumor" if pred > 0.5 else "No Tumor"
    confidence = float(pred)

    return overlay_rgb, f"{label} (confidence = {confidence:.2f})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload MRI Scan"),
    outputs=[
        gr.Image(label="Grad-CAM Output"),
        gr.Text(label="Prediction")
    ],
    title="Brain Tumor Detection with Grad-CAM",
    description="Upload MRI image to detect tumor and visualize Grad-CAM heatmap."
)

interface.launch(share=True)
