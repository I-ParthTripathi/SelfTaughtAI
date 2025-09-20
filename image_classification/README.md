# Image Classifier (Transfer Learning) — quick start

## What this repo contains
- train.py: PyTorch training script using torchvision pretrained models
- app.py: Flask app to serve inference
- Dockerfile: for containerizing the app

## Quick commands (after copying files):
# pip install -r requirements.txt
# python train.py --data_dir ./data --epochs 5 --batch 32
# python app.py  # starts Flask on 0.0.0.0:5000