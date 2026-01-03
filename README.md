# 💳 Credit Card Fraud Detection using Machine Learning

🔍 This project focuses on detecting **fraudulent credit card transactions** using machine learning techniques while addressing the challenge of **severely imbalanced data**.

---

## 🚀 Models Implemented
- 🧠 **Logistic Regression** *(with SMOTE)*
- 🌲 **Random Forest**
  - With SMOTE  
  - Without SMOTE *(using class weighting)*
- ⚡ **XGBoost**
  - With SMOTE  
  - Without SMOTE *(using `scale_pos_weight`)*

---

## 🏆 Best Performing Model
**XGBoost (Without SMOTE, With Class Weighting)** achieved the best balance:

| Metric | Score |
|------|------|
| 🎯 Precision | **0.82** |
| 🔁 Recall | **0.84** |
| ⭐ F1-Score | **0.83** |

✔️ This configuration provided strong generalization and minimized false positives while maintaining high fraud detection capability.

---

## 📊 Dataset
📌 **Kaggle Credit Card Fraud Detection Dataset**  
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> ⚠️ Due to Kaggle licensing restrictions, the dataset is **not included** in this repository.  
> Please download `creditcard.csv` manually and place it inside the `data/` folder.

---

## 🛠️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
