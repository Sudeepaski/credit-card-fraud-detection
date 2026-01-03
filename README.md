# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning techniques.

## Models Used
- Logistic Regression (with SMOTE)
- Random Forest (with and without(with class weighting) SMOTE)
- XGBoost (with and without SMOTE)

## Best Model
XGBoost trained without SMOTE using class weighting achieved the best balance:
- Precision: 0.82
- Recall: 0.84
- F1-score: 0.83

## Dataset
Kaggle Credit Card Fraud Detection Dataset  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## How to Run
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
