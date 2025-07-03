🐶🐱 Paws and Pixels: Cat vs Dog Image Classification

This project involves building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset comes from the Kaggle Dogs vs. Cats competition.

📂 Dataset Overview

Source: Kaggle Dogs vs. Cats DatasetImages: 25,000 labeled training images (cats and dogs)Format: JPEG

🔧 Project Workflow

1. Data Access & Extraction

  Used Kaggle API to download the dataset
  Unzipped training and test data
  Listed and counted total number of training/test images

2. Data Exploration

  Visualized sample images from both classes
  Verified dimensions and color channels
  Plotted image distribution by label (balanced dataset)
  Analyzed individual color channels (R, G, B) using OpenCV

3. Image Preprocessing

  Resized images to (256, 256)
  Normalized pixel values to range [0, 1]
  Augmented training data with rotation, zoom, flips, and shifts
  Used ImageDataGenerator to streamline preprocessing

4. Data Splitting

  Train-Test split (80% train, 20% test)
  Further split train into train and validation sets
  Ensured labels were cast to string format for compatibility

🧠 Model Architecture
  
  Conv2D layers with increasing filters (32, 64, 128)
  MaxPooling2D to reduce dimensionality
  Flatten layer to transition from convolution to dense layers
  Dense(512) with Dropout(0.5) to reduce overfitting
  Final layer: Dense(1, activation='sigmoid') for binary classification

Compilation:

  Optimizer: Adam
  Loss: Binary Crossentropy
  Metrics: Accuracy
  Training Strategy:
  Epochs: 10
  Callbacks: ReduceLROnPlateau, EarlyStopping
  Batch size: 32
  Evaluated model performance using both training and validation generators

📊 Model Evaluation

  Visualized training/validation accuracy and loss
  Final model training accuracy: ~97%
  Final validation accuracy: ~96%
  Evaluated model with:
  Accuracy
  Confusion Matrix
  Classification Report (Precision, Recall, F1)
  ROC Curve (AUC ≈ 0.93)
  Misclassification Analysis:
  Displayed incorrectly classified cat/dog images
  Observed confusion sources visually

🧪 Testing

  Used ImageDataGenerator to normalize test images
  Predicted test labels and compared with ground truth
  Created and visualized confusion matrix
  Generated classification report with accuracy, precision, recall, and F1-score

✅ Summary

  Built and trained a CNN from scratch for binary image classification
  Applied augmentation, dropout, learning rate scheduling, and early stopping to improve performance
  Achieved strong metrics and visualized model behavior on both correct and incorrect predictions

💾 Deliverables

  Final Jupyter Notebook
  CNN model (architecture and weights)
  Evaluation plots (training/validation accuracy, confusion matrix, ROC curve)

👩‍💻 Author

Fatma BozovaEmail: turannfatma@gmail.comLinkedIn: linkedin.com/in/fatma-bozovaGitHub: github.com/FatmaBozova
