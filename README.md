# Sentiment Analysis of E-Commerce User Reviews Using Machine Learning

A Flask-based web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify text into **Positive, Neutral, and Negative** sentiments.

## Project Overview

This project demonstrates an end-to-end sentiment analysis pipeline:

1. Text preprocessing
2. TF-IDF feature extraction
3. Logistic Regression classification
4. Real-time sentiment prediction using Flask

The trained model achieved **78.82% accuracy** on the test split used during training.

## Features

- Positive, Neutral, and Negative sentiment classification
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Logistic Regression machine learning model
- Prediction confidence score
- Interactive Flask web interface
- Real-time sentiment analysis

## Tech Stack

- **Python**
- **Flask**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK**
- **TF-IDF**
- **Logistic Regression**
- **HTML/CSS**

## Project Structure

```text
sentiment-analysis-project/
│
├── app.py
├── train_model.py
├── check_dataset.py
├── requirements.txt
├── model.pkl
├── vectorizer.pkl
│
├── dataset/
│   └── reviews.csv
│
├── templates/
│   └── index.html
│
├── static/
│
└── utils/
    └── preprocess.py
