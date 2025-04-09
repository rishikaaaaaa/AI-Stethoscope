# AI-Stethoscope
AI-powered stethoscope that classifies heart sounds in real time using Raspberry Pi and deep learning.
# 🩺 AI Stethoscope – Real-Time Heart Sound Classification

An intelligent stethoscope system that uses deep learning and signal processing to classify heartbeat sounds as **Normal**, **Murmur**, or **Extrasystolic** in real time using a **Raspberry Pi 4**.

---

## 📌 Project Description

This project implements an **AI-powered stethoscope** that records live heart sounds using a microphone connected to a Raspberry Pi and classifies them using a lightweight **TensorFlow Lite** model. It leverages **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and runs inference on-device, providing fast, accurate, and interpretable results through the terminal or an optional web interface.

---

## 🚀 Features

- Real-time heartbeat audio recording using USB microphone  
- Signal amplification and noise reduction  
- MFCC feature extraction using `librosa`  
- CNN-based deep learning model trained on heartbeat datasets  
- Deployment via TensorFlow Lite on Raspberry Pi  
- Web-based GUI (optional Flask app)  
- Spectrogram and waveform visualization for debugging  

---


## 🛠 Tech Stack

- **Python 3.11**
- **Raspberry Pi OS (Bookworm)**
- `TensorFlow` / `tflite_runtime`
- `sounddevice`, `librosa`, `matplotlib`, `numpy`
- RealVNC (for remote display)

---

## 🧠 Model Overview

| Layer              | Type             | Output Shape     | Activation |
|-------------------|------------------|------------------|------------|
| Input              | (13, 128, 1)     | -                | -          |
| Conv2D + Pooling   | Feature Extraction | -              | ReLU       |
| Dropout            | Regularization   | -                | -          |
| Dense              | Fully Connected  | 64               | ReLU       |
| Output             | Dense            | 3                | Softmax    |

- Model trained on Kaggle Heart Sound Dataset
- Converted to `.tflite` for deployment
