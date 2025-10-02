Snapp Sentiment Analysis

Overview
This project implements a sentiment analysis model for Persian comments, specifically targeting user reviews from services like Snapp (a ride-hailing and food delivery app). The model classifies comments as HAPPY (positive) or SAD (negative) using a fine-tuned BERT-based model. It leverages the Hugging Face Transformers library for model training and inference.
The core of the project is a Jupyter Notebook (Snapp(1).ipynb) that handles data loading, preprocessing, model fine-tuning, evaluation, and sample predictions.
Key Features

Dataset: Uses the "SBU-NLPClass/Sentiment-Analysis" dataset from Hugging Face, which includes Persian comments labeled as HAPPY or SAD.
Model: Fine-tuned on "sharif-dal/dal-bert" (a Persian BERT variant) for sequence classification.
Metrics: Evaluates using Accuracy and Weighted F1-score.
Training: Utilizes Hugging Face's Trainer API with mixed-precision training (FP16) for efficiency.
Inference: Simple API-like usage for predicting sentiment on new text.

This project can be extended for multi-class sentiment (e.g., adding MIXED) or applied to other Persian NLP tasks.
Requirements

Python 3.8+
Jupyter Notebook or Google Colab for running the notebook
Key libraries (install via pip install -r requirements.txt):

transformers (Hugging Face)
datasets (Hugging Face)
torch (PyTorch)
scikit-learn (for metrics)
accelerate (for distributed training, optional)
peft (for LoRA, if using parameter-efficient fine-tuning)
Training the Model

The notebook uses Hugging Face's Trainer with the following key args:

Batch size: 64 (train/eval)
Epochs: 2
Learning rate: 2e-5
Weight decay: 0.01


Run trainer.train() to start fine-tuning. Model checkpoints are saved in ./results.

Evaluation

Computes Accuracy and F1-score on the validation set.
Example results from training:

Epoch 1: Accuracy ~88.14%, F1 ~88.11%
Epoch 2: Accuracy ~88.17%, F1 ~88.16%

Dataset

Source: Hugging Face Datasets ("SBU-NLPClass/Sentiment-Analysis")
Splits: Train (52,110 examples), Validation (8,337), Test (9,033)
Labels: HAPPY (0), SAD (1)
Preprocessing: Tokenized with max length 128, padded to uniform size.

Model Details

Base Model: "sharif-dal/dal-bert"
Fine-Tuning: Sequence classification head with 2 labels.
Optional: LoRA (Low-Rank Adaptation) for efficient fine-tuning (commented in notebook).
Hardware: Trained on GPU (e.g., T4 in Colab) with FP16 for speed.
