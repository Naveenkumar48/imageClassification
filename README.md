# 🐶🐱 Dog vs Cat Image Classifier using CNN

A deep learning project to automatically classify images of dogs and cats using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## Project Overview

This project demonstrates an end-to-end workflow of:

- Downloading and preparing image data (from Kaggle).
- Building and training a CNN for binary classification.
- Evaluating performance with loss and accuracy metrics.
- Predicting and visualizing the classification of new images.

---

##Key Features

✅ Download dataset automatically via Kaggle API  
✅ Preprocess and normalize images  
✅ Build a custom CNN with multiple convolutional and dense layers  
✅ Visualize training and validation metrics  
✅ Predict on custom test images  

---

## Model Architecture

- **Input:** 256x256 RGB images
- **Layers:**
  - 3 × Conv2D layers with ReLU activation
  - MaxPooling and BatchNormalization after each Conv2D
  - Flatten + Dense layers with Dropout regularization
  - Sigmoid output for binary classification
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Kaggle API
- Jupyter Notebook / Google Colab

---

## 📈 Training & Evaluation

The model is trained over **5 epochs** on the Kaggle "Dogs vs Cats" dataset.  

### Sample Training Results:
- **Training Accuracy:** ~85% (depending on epochs and dataset splits)
- **Validation Accuracy:** ~80%

### Loss and Accuracy Graphs:
![Sample Loss and Accuracy](sample_output.png)

---

## 🖼️ Example Prediction

Test image loaded with OpenCV:

![Test Image](example_dog.jpg)

Model output: **Dog**

