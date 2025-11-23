# Brain-Tumor-Detection

Brain Tumor Detection using Customized CNN with Explainable AI (SHAP, LIME, Grad-CAM)
Replicated exactly from the Heliyon 2024 research paper: “Utilizing customized CNN for brain tumor prediction with explainable AI”.

🧠 Brain Tumor Detection Using CNN + XAI

This project implements a Customized Convolutional Neural Network (CNN) for brain tumor classification using MRI images, along with Explainable AI (XAI) techniques — SHAP, LIME, and Grad-CAM — to explain predictions.

The entire methodology, architecture, dataset, preprocessing, and evaluation strictly follow the research paper.

📂 Dataset

The project uses the publicly available BR35H brain MRI dataset:

🔗 Dataset Link:
https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

Dataset structure:

yes/       → 1500 tumor images  
no/        → 1500 normal images  
pred/      → 60 test images  


Images are resized to 150 × 150 × 3, normalized to [0,1].

🏗️ Project Pipeline
✔ 1. Dataset Upload & Extraction (Colab)

Dataset is uploaded manually and extracted into the /data folder.

✔ 2. Data Preprocessing

Image resizing → (150×150×3)

Normalization → pixel/255

Label encoding

Train/Validation split → 90% / 10%

🧱 Customized CNN Architecture (from the research paper)

The model uses 4 convolutional blocks with increasing filters:

Layer	Filters	Kernel	Activation
Conv2D	32	3×3	ReLU
Conv2D	64	3×3	ReLU
Conv2D	128	3×3	ReLU
Conv2D	256	3×3	ReLU

Followed by:

Flatten

Dense(128) + ReLU

Dense(64) + ReLU

Dropout(0.5)

Dense(1) + Sigmoid

Total Parameters: ~1.51 million
Loss: Binary Cross-Entropy
Optimizer: Adam

This is the exact model from the research paper.

📊 Model Performance (Matches Research Paper)
Metric	Value
Training Accuracy	100%
Validation Accuracy	98–99%
Precision	~98.5%
Recall	~98.5%
F1-Score	~98.5%

Confusion matrix indicates:

Very few false positives

Very few false negatives

🔍 Explainable AI (XAI)
✔ SHAP

Highlights which image regions influence the model toward:

Tumor (red regions)

Non-tumor (blue regions)

Uses DeepExplainer.

✔ LIME

Generates yellow superpixel regions showing which parts of the MRI contributed to prediction.

✔ Grad-CAM

Creates heatmaps showing where the CNN is focusing in the MRI image.
Red areas = high importance
Blue areas = low importance

📈 Training Graphs

Includes:

Training vs Validation Accuracy

Training vs Validation Loss

These help verify overfitting or convergence behavior.

🖥 Running the Project
1. Clone this repository
git clone https://github.com/your-username/brain-tumor-detection-xai.git
cd brain-tumor-detection-xai

2. Open the Colab Notebook

Upload the notebook or click:
(your own link once you add it)

3. Upload the Kaggle Dataset to Colab

Use:

from google.colab import files
files.upload()


Upload brain-tumor-detection.zip.

4. Extract the dataset
import zipfile
with zipfile.ZipFile("brain-tumor-detection.zip", "r") as z:
    z.extractall("data")

5. Run all cells

The notebook will:

Preprocess images

Train CNN

Plot metrics

Generate XAI visualizations

🧪 Example Outputs Included

SHAP overlays

LIME superpixel explanations

Grad-CAM heatmaps

Confusion matrix

Metric tables

🔮 Future Work

As recommended in the research paper:

Multi-class tumor classification

Experiment with Transformer models

Deployment as Flask/Streamlit webapp

Integration with clinical PACS systems

👨‍🎓 Author

Tanisha Gupta
B.Tech CSE (Data Science)
AI/ML Project — Brain MRI Tumor Detection
2024–25
