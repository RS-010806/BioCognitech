# Breast Cancer Image Recognition using AI/ML

# Overview

This repository contains an AI/ML-based breast cancer image recognition model. A mammogram is an x-ray picture of the breast. It can be used to check for breast cancer in women who have no signs or symptoms of the disease. It can also be used if you have a lump or other sign of breast cancer. Now while its difficult to figure out for physicians by seeing only images of x-ray that weather the tumor is toxic or not training a machine learning model for the identification of tumour can be of great help. The model leverages ResNet, CNNs, and an advanced data augmentation pipeline to classify medical images with high precision.

![image](https://github.com/user-attachments/assets/0cd0c0ba-80d5-45eb-b4f3-2955f4c6e1e7)

# Features

- Deep Learning Model: Utilizes ResNet18 architecture for feature extraction and classification.

- Advanced Data Augmentation: Includes resizing, flipping, rotation, and color jittering to enhance training.

- Optimized Training: Implements Adam optimizer with learning rate scheduling.

- Evaluation Metrics: Tracks accuracy, precision, recall, and F1-score.

- GPU Acceleration: Supports CUDA for enhanced training efficiency.

# Dataset

The model is trained on a breast cancer image dataset using PyTorch's ImageFolder:

Train Directory: train/

Test Directory: test/

# Installation Prerequisites

Ensure you have Python installed (recommended version: 3.8+). Install dependencies using:

pip install -r requirements.txt

# Training the Model

To train the model, execute:

python train.py

# Evaluation

To evaluate the trained model, run:

python evaluate.py

# Predicting on New Images

To classify new images using the trained model, use:

python predict.py --image path/to/image.jpg

# Model Details

Architecture: DenseNet (pre-trained on ImageNet, fine-tuned for breast cancer recognition).

Loss Function: CrossEntropyLoss.

Optimizer: Adam with learning rate scheduling.

Batch Size: 32.

Epochs: 50.

# Team

This project was developed by:
Rishi Shah
Saksham Chaabra
Shivaay Dhondiyaal
Shubhank Gupta

# Achievements

2nd Place at Research Forge Hackathon | Invictus, DTU 🏆

Only First-Year Team competing in the ML track 🎯

Integration of AI and Biotechnology for impactful healthcare solutions

# Future Work

Improve model interpretability and robustness.

Experiment with Transformer-based architectures for enhanced accuracy.

Develop a web-based application for real-time diagnostics.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.

# Contributing

Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.

# Contact

For any queries, please reach out to rishishah010806@gmail.com or connect via LinkedIn: https://www.linkedin.com/in/r1shi-shah/.


