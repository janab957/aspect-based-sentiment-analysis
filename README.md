# Aspect-Based Sentiment Analysis on Customer Reviews

## Overview
An NLP-based customer feedback analysis system that identifies specific aspects mentioned in customer reviews and classifies their sentiments as positive, negative, or neutral.

The project compares traditional machine learning methods with transformer-based models to evaluate their effectiveness in multi-label aspect-based sentiment analysis (ABSA).

## Dataset
- **FABSA Dataset:** 10,574 customer reviews from multiple industries.
- **12 aspect categories** with positive, negative, and neutral sentiment labels.
- Each review can contain multiple aspects with different sentiments.

## Models Implemented
- Logistic Regression (BoW and TF-IDF)
- Support Vector Machine (BoW and TF-IDF)
- RoBERTa
- DeBERTa

## Model Performance

| Model | F1-score |
|---|---|
| Logistic Regression (BoW) | 65.2% |
| Logistic Regression (TF-IDF) | 57.2% |
| SVM (BoW) | 65.7% |
| SVM (TF-IDF) | 64.7% |
| RoBERTa | 77.7% |
| **DeBERTa (Optimized)** | **81.3%** |

DeBERTa achieved the strongest overall F1-score, with a Hamming accuracy of 98.1% following threshold optimization.

## Application Features
- Customer review sentiment analysis
- Multi-aspect identification and classification
- Aspect-level sentiment and confidence scores
- Overall sentiment classification (Positive, Negative, Neutral, or Mixed)
- Executive summary and actionable recommendations
- User-friendly graphical interface

## Technologies
Python, Scikit-learn, NLTK, Hugging Face Transformers, RoBERTa, and DeBERTa.

