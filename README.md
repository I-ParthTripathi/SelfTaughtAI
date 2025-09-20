# 🧠 Self-Taught AI Engineer Portfolio

Welcome to my AI/ML portfolio! 🚀  
I am a **self-taught AI/ML enthusiast** who learned from **free resources (YouTube, MOOCs, GitHub)** and built practical projects to strengthen my fundamentals.  
This repository contains my **resume + projects** that showcase my journey into AI engineering.

---

## 📂 Repository Structure

```
SelfTaughtAI/
│── resume.md                # My resume
│
├── image_classification/     # CNN project (PyTorch + Flask + Docker)
├── house_price_pipeline/     # End-to-end ML pipeline (Sklearn + FastAPI)
└── nlp_sentiment_hf/         # NLP project (Hugging Face Transformers)
```

---

## 🏗️ Projects

### 🔹 1. [Image Classification (CNN)](./image_classification)
- Built a **Convolutional Neural Network (PyTorch)** to classify images (CIFAR-10 dataset).  
- Added a **Flask API** for serving predictions.  
- Packaged with **Docker** for easy deployment.  
- **Tech:** Python, PyTorch, Flask, Docker  

---

### 🔹 2. [House Price Prediction Pipeline](./house_price_pipeline)
- Developed a **regression model** to predict house prices.  
- Built a **scikit-learn pipeline** with preprocessing + model training.  
- Served using **FastAPI** with real-time prediction endpoints.  
- **Tech:** Python, scikit-learn, FastAPI  

---

### 🔹 3. [NLP Sentiment Analysis](./nlp_sentiment_hf)
- Fine-tuned a **Hugging Face Transformer** model for sentiment analysis.  
- Trained on a custom text dataset.  
- Exposed a lightweight **inference API**.  
- **Tech:** Python, Hugging Face Transformers  

---

## 📜 Resume
My detailed resume is available here → [resume.md](./resume.md)

---

## 🚀 How to Run Projects Locally

Clone this repo:
```bash
git clone https://github.com/YOURUSERNAME/SelfTaughtAI.git
cd SelfTaughtAI
```

Go inside any project folder and follow the instructions in its `README.md`.  
Example:
```bash
cd image_classification
pip install -r requirements.txt
python train.py
```

---

## 🎯 Highlights
✅ Learned AI/ML from **YouTube & free resources**  
✅ Built **end-to-end projects** (modeling → serving → deployment)  
✅ Hands-on with **PyTorch, scikit-learn, Hugging Face, Flask, FastAPI, Docker**  
✅ Open-source mindset — sharing everything here for others to learn  

---

## 🤝 Connect With Me
- 🌐 Portfolio: *[Add your portfolio link if any]*  
- 💼 LinkedIn: *[Your LinkedIn]*  
- 📧 Email: *[Your Email]*  

---

🔥 This repo is a reflection of my **self-learning journey** in AI/ML — proving that you don’t need a fancy degree to build great things!

---

# 📁 image_classification/README.md

# 🖼️ Image Classification (CNN)

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10 dataset**.  
The trained model is served with a **Flask API**, and a **Dockerfile** is included for containerized deployment.

---

## 🚀 Features
- CNN built in **PyTorch**
- Flask API for inference
- Dockerized for deployment
- Trained on CIFAR-10 dataset

---

## 🛠️ Installation & Usage
```bash
cd image_classification
pip install -r requirements.txt
python train.py   # Train the model
python app.py     # Run the Flask server
```

API Example:
```bash
POST /predict
Content-Type: multipart/form-data
file: <image_file>
```

---

## 📦 Tech Stack
- Python
- PyTorch
- Flask
- Docker

---

# 📁 house_price_pipeline/README.md

# 🏠 House Price Prediction Pipeline

An **end-to-end regression pipeline** for predicting house prices using **scikit-learn**.  
Includes preprocessing, model training, and real-time inference served with **FastAPI**.

---

## 🚀 Features
- Regression model (RandomForestRegressor)
- Preprocessing with scikit-learn pipelines
- FastAPI service for predictions

---

## 🛠️ Installation & Usage
```bash
cd house_price_pipeline
pip install -r requirements.txt
python pipeline.py    # Train & save model
python serve.py       # Start FastAPI server
```

API Example:
```bash
POST /predict
{
  "area": 2000,
  "bedrooms": 3,
  "bathrooms": 2
}
```

---

## 📦 Tech Stack
- Python
- scikit-learn
- FastAPI

---

# 📁 nlp_sentiment_hf/README.md

# 💬 NLP Sentiment Analysis

This project fine-tunes a **Hugging Face Transformer model** for sentiment analysis (positive/negative).  
It is trained on a custom dataset and provides a simple **inference API**.

---

## 🚀 Features
- Fine-tuned Hugging Face Transformer
- Preprocessing & tokenization pipeline
- Lightweight inference script

---

## 🛠️ Installation & Usage
```bash
cd nlp_sentiment_hf
pip install -r requirements.txt
python inference.py "I love this project!"
```

Output Example:
```
Sentiment: Positive
Confidence: 0.92
```

---

## 📦 Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
