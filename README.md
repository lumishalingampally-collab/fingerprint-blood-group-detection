# fingerprint-blood-group-detection
Deep learning-based blood group classification from fingerprint images using ResNet-50, PyTorch, and Flask.

Absolutely. Based on your uploaded mini-project report, your project is **Fingerprint-Based Blood Group Detection using Deep Learning**, using **ResNet-50, PyTorch, Flask, and an 8-class fingerprint dataset**. 


## 📌 Project Overview

Blood group identification is traditionally performed using blood samples and laboratory testing. This project explores an AI-based, non-invasive approach for preliminary blood group classification using fingerprint image patterns.

The system uses a pretrained **ResNet-50** deep learning model and fine-tunes it to classify fingerprint images into eight blood group categories:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

The trained model is integrated with a **Flask web application**, where users can upload a fingerprint image and receive the predicted blood group.

The project was developed using **Python, PyTorch, Torchvision, Google Colab, Flask, HTML, CSS, and JavaScript**.

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop a CNN-based fingerprint image classification model.
2. To investigate the possibility of predicting blood groups from fingerprint patterns.
3. To use transfer learning with the ResNet-50 architecture.
4. To preprocess and augment fingerprint images for deep learning.
5. To train the model for eight blood group classes.
6. To evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. To integrate the trained model into a Flask web application.
8. To provide a simple interface for uploading fingerprint images and displaying predictions.

---

## ✨ Features

- 🖐️ Fingerprint image upload
- 🤖 ResNet-50 based deep learning model
- 🧠 Transfer learning
- 🔄 Image preprocessing and augmentation
- 🩸 Eight blood group classifications
- 🌐 Flask-based web interface
- 📊 Model evaluation using multiple metrics
- 📈 Accuracy and loss visualization
- 🔲 Confusion matrix analysis
- ⚡ Fast prediction using a saved `.pth` model
- ☁️ Google Colab compatible training environment

---

## 🏗️ System Architecture

The system follows the following pipeline:

```text
                 ┌──────────────────────┐
                 │ Fingerprint Dataset  │
                 │    8000 Images       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 │ Resize / Normalize   │
                 │ Data Augmentation    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      ResNet-50       │
                 │  Transfer Learning   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   8-Class Output     │
                 │ A+ A- B+ B- AB+ AB- │
                 │       O+ O-          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Trained .pth Model   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Flask Web App      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Upload Fingerprint   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Predicted Blood Group│
                 └──────────────────────┘
````

---

## 📊 Dataset

The project uses a dataset containing **8,000 fingerprint images** distributed across eight blood group classes.

Each blood group contains approximately **1,000 images**, making the dataset balanced for multi-class classification.

### Blood Group Classes

| Class | Description |
| ----- | ----------- |
| A+    | A Positive  |
| A-    | A Negative  |
| B+    | B Positive  |
| B-    | B Negative  |
| AB+   | AB Positive |
| AB-   | AB Negative |
| O+    | O Positive  |
| O-    | O Negative  |

The fingerprint images are organized into separate folders according to their blood group labels.

Example:

```text
dataset_blood_group/
│
├── A+/
├── A-/
├── AB+/
├── AB-/
├── B+/
├── B-/
├── O+/
└── O-/
```

The dataset is loaded using PyTorch's `ImageFolder` utility.

---

## 🧹 Data Preprocessing

Before training, the fingerprint images undergo preprocessing.

### Training Transformations

The training pipeline includes:

* Resizing
* Random horizontal flipping
* Random rotation
* Brightness adjustment
* Contrast adjustment
* Conversion to tensor
* Normalization

These transformations increase the variety of training samples and help improve model generalization.

### Validation Transformations

Validation images undergo:

* Resizing
* Conversion to tensor
* Normalization

No augmentation is applied to the validation images.

---

## 🔀 Dataset Split

The complete dataset is divided into:

* **80% Training**
* **20% Validation**

A batch size of **32** is used during training and validation.

The training data is shuffled to improve learning.

---

## 🧠 Deep Learning Model

### ResNet-50

The project uses **ResNet-50**, a deep Convolutional Neural Network architecture based on residual learning.

A pretrained ResNet-50 model is used as the starting point.

Instead of training the complete network from scratch:

* Most pretrained layers are frozen.
* The final convolutional block (`layer4`) is made trainable.
* The final fully connected (`fc`) layer is replaced.
* The new output layer produces predictions for eight blood group classes.

```text
Pretrained ResNet-50
        │
        ▼
Feature Extraction
        │
        ▼
Layer 4 Fine-Tuning
        │
        ▼
Fully Connected Layer
        │
        ▼
8 Blood Group Classes
```

---

## 🔄 Transfer Learning

Transfer learning allows the model to use features learned from a large image dataset and adapt them to the fingerprint classification problem.

The pretrained ResNet-50 provides useful general image features while the final layers are fine-tuned using the fingerprint dataset.

This approach reduces training requirements compared with training a deep CNN entirely from scratch.

---

## ⚙️ Training Configuration

| Parameter              | Configuration    |
| ---------------------- | ---------------- |
| Model                  | ResNet-50        |
| Framework              | PyTorch          |
| Dataset Size           | 8,000 images     |
| Number of Classes      | 8                |
| Train/Validation Split | 80/20            |
| Batch Size             | 32               |
| Loss Function          | CrossEntropyLoss |
| Optimizer              | Adam             |
| Scheduler              | StepLR           |
| Training Epochs        | 15               |
| Input Size             | 224 × 224        |
| Model Format           | `.pth`           |
| Environment            | Google Colab     |

> Note: The project report's abstract mentions 10 epochs, while the implementation and result-analysis sections describe training for 15 epochs. This README follows the detailed implementation/results sections and uses 15 epochs.

---

## 📉 Loss Function

The project uses:

```python
CrossEntropyLoss()
```

Cross-entropy loss is suitable for multi-class classification because the model needs to select one class from eight possible blood groups.

---

## 🚀 Optimizer

The Adam optimizer is used for training.

```python
optimizer = torch.optim.Adam(...)
```

Adam provides adaptive learning-rate updates and is commonly used for deep learning optimization.

Only the trainable parameters of the model are optimized.

---

## 📅 Learning Rate Scheduler

A StepLR scheduler is used to reduce the learning rate during training.

The learning rate is reduced by a factor of **0.1 every 5 epochs**.

This helps the model converge more smoothly during later stages of training.

---

## 🏋️ Model Training

The training process consists of the following steps:

1. Load the fingerprint dataset.
2. Apply training transformations.
3. Split the dataset into training and validation sets.
4. Create PyTorch DataLoaders.
5. Load pretrained ResNet-50.
6. Freeze the required layers.
7. Replace the final fully connected layer.
8. Define CrossEntropyLoss.
9. Define Adam optimizer.
10. Train the model.
11. Validate after every epoch.
12. Track training and validation accuracy.
13. Save the best-performing model.

The best model is saved based on validation accuracy.

---

## 💾 Model Saving

The trained model is saved in PyTorch `.pth` format.

Example:

```text
best_blood_group_model.pth
```

The saved model can later be loaded for prediction without retraining.

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics provide a more detailed understanding of model performance than accuracy alone.

---

## 📈 Results

The project report reports approximately:

### Test Accuracy

**97.81%**

The result was obtained on the test set described in the project report.

The report also describes validation accuracy stabilizing around **88–89%**, while training accuracy approached approximately **99.9%**.

```text
Training Accuracy    ≈ 99.9%
Validation Accuracy  ≈ 88–89%
Test Accuracy        ≈ 97.81%
```

---

## 📋 Classification Results

The reported classification metrics are:

| Blood Group | Precision | Recall | F1-Score |
| ----------- | --------- | ------ | -------- |
| A+          | 0.99      | 0.98   | 0.98     |
| A-          | 0.96      | 0.95   | 0.95     |
| AB+         | 0.98      | 0.97   | 0.98     |
| AB-         | 0.96      | 0.97   | 0.97     |
| B+          | 0.97      | 0.98   | 0.97     |
| B-          | 0.98      | 1.00   | 0.99     |
| O+          | 0.96      | 0.97   | 0.97     |
| O-          | 0.98      | 0.97   | 0.97     |

Overall reported metrics are approximately:

```text
Accuracy     : 0.97
Macro Avg    : 0.97
Weighted Avg : 0.97
```

---

## 🔲 Confusion Matrix

A confusion matrix is used to visualize the classification performance of the eight blood group classes.

Values along the diagonal represent correctly classified samples, while off-diagonal values represent misclassifications.

The project report notes minor confusion between visually similar fingerprint classes.

---

## 🌐 Web Application

The trained model is integrated into a Flask web application.

The web interface provides:

1. Fingerprint image upload.
2. Image preprocessing.
3. Model inference.
4. Blood group prediction.
5. Display of the uploaded fingerprint.
6. Display of the predicted blood group.

### Interface

The application uses a simple black-themed interface with white text and a centered upload form.

```text
┌────────────────────────────────────────────┐
│                                            │
│   FINGERPRINT BASED BLOOD GROUP DETECTION  │
│                                            │
│        ┌─────────────────────────┐         │
│        │   Choose Fingerprint    │         │
│        └─────────────────────────┘         │
│                                            │
│              [ Predict ]                   │
│                                            │
│        Predicted Blood Group: O+           │
│                                            │
│             Fingerprint Image              │
│                                            │
└────────────────────────────────────────────┘
```

---

## 🔄 Prediction Workflow

When a user uploads an image:

```text
User Uploads Fingerprint
          │
          ▼
Image Validation
          │
          ▼
Resize to 224 × 224
          │
          ▼
Convert to RGB
          │
          ▼
Convert to Tensor
          │
          ▼
Normalize Image
          │
          ▼
ResNet-50
          │
          ▼
Model Output
          │
          ▼
Highest Probability Class
          │
          ▼
Predicted Blood Group
```

---

## 🖼️ Image Preprocessing During Inference

The uploaded image is processed using the same general preprocessing pipeline used during training.

The inference pipeline includes:

```text
Resize → RGB Conversion → Tensor Conversion → Normalization
```

The input image is resized to:

```text
224 × 224 pixels
```

This matches the expected input dimensions of the ResNet-50 model.

---

## 🧮 Prediction

The ResNet-50 model generates output scores for all eight blood group classes.

Softmax can be applied to convert the model's output logits into class probabilities.

The class with the highest probability is selected as the predicted blood group.

Example:

```text
A+   → 0.02
A-   → 0.01
B+   → 0.04
B-   → 0.03
AB+  → 0.05
AB-  → 0.02
O+   → 0.81  ← Prediction
O-   → 0.02
```

Result:

```text
Predicted Blood Group: O+
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision
* ResNet-50
* Transfer Learning

### Data Processing

* PIL
* NumPy
* Torchvision Transforms

### Machine Learning Evaluation

* Scikit-learn
* Classification Report
* Confusion Matrix

### Visualization

* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Development Environment

* Google Colab
* Google Drive

### Deployment / Tunneling

* PyNgrok / ngrok

---


## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fingerprint-blood-group-detection.git
```

Move into the project directory:

```bash
cd fingerprint-blood-group-detection
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📄 requirements.txt

Example dependencies:

```text
torch
torchvision
flask
numpy
pillow
matplotlib
seaborn
scikit-learn
tqdm
pyngrok
```

---

## ▶️ Running the Application

After installing the dependencies and placing the trained model in the correct directory:

```bash
python app.py
```

The Flask application will start locally.

Open the displayed local address in a web browser.

---

## 🧪 Using the Application

1. Open the web application.
2. Click **Choose File**.
3. Select a compatible fingerprint image.
4. Click **Predict**.
5. The trained ResNet-50 model processes the image.
6. The predicted blood group is displayed.

---

## ☁️ Google Colab

Model training can be performed using Google Colab.

The project uses Google Drive to store:

* Dataset
* Trained model
* Prediction images

Example dataset location:

```text
/content/drive/MyDrive/dataset_blood_group
```

The Google Drive can be mounted using:

```python
from google.colab import drive

drive.mount('/content/drive')
```

---

## ⚠️ Limitations

The current prototype has several limitations:

* It requires a compatible fingerprint image.
* Prediction quality depends on fingerprint image quality.
* The system is limited to the eight trained blood group classes.
* It does not currently support real-time fingerprint scanner hardware.
* The model depends on the characteristics of the training dataset.
* The system is not intended for critical medical decisions.
* Clinical/laboratory verification is required for actual blood group determination.

---

## 🔮 Future Scope

Future improvements could include:

### 1. Larger Dataset

Train the model using a larger and more diverse fingerprint dataset.

### 2. Real-Time Fingerprint Scanner

Integrate fingerprint scanner hardware instead of relying only on uploaded images.

### 3. Mobile Application

Deploy the model as an Android/iOS application.

### 4. Lightweight Models

Use lightweight architectures such as:

* MobileNet
* Quantized ResNet
* Other efficient CNN architectures

This could improve deployment on mobile and edge devices.

### 5. Improved Generalization

Use more diverse data from different populations, fingerprint qualities, and acquisition devices.

### 6. Additional Biometric Features

Future versions could investigate combining fingerprint features with other biometric information.

### 7. Healthcare Integration

A future prototype could explore integration with digital health records, subject to appropriate privacy, security, validation, and regulatory requirements.

---

## 🎓 Academic Applications

This project demonstrates the practical application of:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Transfer Learning
* Image Classification
* Healthcare AI
* Biometric Image Analysis
* Web Application Development

It can be used as an academic demonstration of how a trained deep learning model can be integrated into a web application.

---

## 📚 References

The project report references research related to fingerprint-based blood group identification, deep learning, biometrics, and ResNet architectures.

Important references include:

1. K. Meena and V. R. R. Kumari, "Fingerprint Based Blood Group Identification using Image Processing Techniques," International Journal of Engineering and Technology.

2. S. S. Shinde and D. S. Waghmare, "Blood Group Detection using Fingerprint: A Machine Learning Approach," IRJET, 2020.

3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.

4. N. V. Rao, M. Arathi, A. S. Swathi, S. R. Rajesh, "Analysis of Biometric Fingerprint Patterns in Relation to ABO Blood Group," Journal of Clinical and Diagnostic Research.

5. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS.

---

## 📜 License

This project was developed for academic and educational purposes.

You may adapt and extend the project for learning and research purposes while appropriately acknowledging the original work.

---
#OUTPUT 
<img width="837" height="637" alt="Screenshot 2026-09-30 163810" src="https://github.com/user-attachments/assets/21a13ebd-5031-4ac7-9896-b8d5b577067a" />
<img width="877" height="610" alt="Screenshot 2026-09-30 163729" src="https://github.com/user-attachments/assets/704fa671-27fa-4b43-917c-c2361429c956" />
<img width="685" height="463" alt="Screenshot 2026-09-30 163743" src="https://github.com/user-attachments/assets/282a0eba-01a7-49d3-b5cf-420f3d4643f7" />
<img width="675" height="218" alt="Screenshot 2026-09-30 163752" src="https://github.com/user-attachments/assets/d8d5a7ce-9df6-45fd-b5b9-94a59ac78571" />
<img width="856" height="352" alt="Screenshot 2026-09-30 163712" src="https://github.com/user-attachments/assets/92e381fc-ba88-4244-9a66-6ddb1262cb5f" />




