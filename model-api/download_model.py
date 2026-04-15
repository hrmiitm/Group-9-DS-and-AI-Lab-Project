"""
download_model.py
Run once at Docker build time to pre-cache model weights into the image layer.
This eliminates cold-start download delay when the container launches.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "aditya963/fraud-job-classifier"

print(f"Downloading tokenizer: {MODEL_ID}")
AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

print(f"Downloading model weights: {MODEL_ID}")
AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

print("Model cached successfully.")
