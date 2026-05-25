# Food Image Classification using Deep Learning

## 🚀 Live Demo

Project live at:

(https://image-classification-food101.onrender.com/)

Try uploading an image, dragging and dropping one, or providing an image URL to classify among:

- Pizza 🍕
- Steak 🥩
- Sushi 🍣


## Project Overview

This project is a Computer Vision based Image Classification web application built using PyTorch and Flask.

The model is a custom Convolutional Neural Network (CNN) trained to classify food images into three categories:

- Pizza 🍕
- Steak 🥩
- Sushi 🍣

The web application allows users to:

- Upload an image
- Drag and drop an image
- Provide an image URL
- Predict the food category instantly

---

## Dataset

The model was trained using selected classes from the Food101 dataset.

Classes used:

- Pizza
- Steak
- Sushi

Only these three classes from the original dataset were used for training.

Dataset source:

https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

---

## Model Architecture

Custom CNN architecture built using PyTorch:

Input Image
↓
Conv2D
↓
ReLU
↓
Conv2D
↓
ReLU
↓
MaxPool
↓
Conv2D
↓
ReLU
↓
Conv2D
↓
ReLU
↓
MaxPool
↓
Conv2D
↓
ReLU
↓
Conv2D
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Fully Connected Layer
↓
Output (3 Classes)

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
- HTML
- CSS
- JavaScript
- PIL
- NumPy
- Matplotlib

---

## Project Structure

```text
image_classifier_web/

├── train.py
├── predict.py
├── app.py
├── model.pth
├── class_names.txt
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── uploads/


### to download and try it , follow these instructions 
#Installation

#Clone repository:

git clone https://github.com/YOUR_USERNAME/image-classifier-web.git

# Move into project directory:

cd image-classifier-web

# Install dependencies:

pip install -r requirements.txt

#Run application:

python app.py

#Open browser:

http://127.0.0.1:5000

### FUTURE IMPROVEMENTS 
 Add entire dataset for 
 - Pizza 🍕
- Steak 🥩
- Sushi 🍣
