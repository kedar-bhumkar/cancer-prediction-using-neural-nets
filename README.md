---
title: Cancer Risk Prediction Model
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
license: mit
---

# Cancer Risk Prediction Model

This Gradio web application predicts cancer risk levels (Low, Medium, High) based on patient characteristics using a neural network model.

## Model Details

The model is a neural network trained on cancer patient data with the following features:
- Input: 23 patient characteristics
- Output: Risk classification (Low, Medium, High)
- Architecture: Multi-layer perceptron

## Usage

1. Adjust the sliders to input patient characteristics
2. The model will automatically predict the risk level
3. Example cases are provided for reference

## Technical Stack

- PyTorch: Neural network framework
- Gradio: Web interface
- Pandas & NumPy: Data processing
- Scikit-learn: Data preprocessing

## Dataset

The model uses the Cancer Patient Data Sets, which includes various patient characteristics and their corresponding risk levels.

## License

This project is licensed under the MIT License.