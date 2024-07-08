# LLM-fraud-detection
Synthetically generated fraud detection LLM project for internal and external demos

# Credit Card Fraud Detection

This repository contains code for detecting credit card fraud using both traditional machine learning models and Large Language Models (LLMs).

## Directory Structure
- `data/`: Contains sample credit card transaction data.
- `models/`: Contains the Python scripts for training and using machine learning models.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and model training.
- `utils/`: Contains utility scripts for data preprocessing and feature engineering.

```
credit-card-fraud-detection/
│
├── data/
│   ├── sample_transactions.csv
│
├── models/
│   ├── llm_fraud_detection.py
│   ├── traditional_ml_models.py
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│
├── requirements.txt
│
├── README.md
│
└── utils/
    ├── data_preprocessing.py
    ├── feature_engineering.py
```

## Requirements
To install the required libraries, run:
```bash
pip install -r requirements.txt
