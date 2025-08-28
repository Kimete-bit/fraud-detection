title: Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
---

# Fraud Detection – PaySim Demo  

🚀 [**Live Demo on Hugging Face**](https://huggingface.co/spaces/Kimete/fraud-detection)

LightGBM pipeline (OneHot on type) trained on PaySim dataset.  
This project demonstrates data preprocessing, model training, evaluation, and deployment with Gradio on Hugging Face Spaces.


# Fraud Detection – PaySim Demo

🚀 **Demo**: This Space hosts a machine learning pipeline for **fraud detection** using the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).  
It demonstrates a full ML workflow: **data preprocessing, model training, pipeline saving, and cloud deployment** with Gradio + Hugging Face Spaces.

---

## 1. Project Overview
Fraud detection is a classic challenge in financial transactions, especially when the data is highly imbalanced (fraud cases are very rare).

This project includes:
- Exploratory Data Analysis (EDA) on PaySim dataset
- Class imbalance handling (SMOTE / class_weight)
- Training models: Logistic Regression, LightGBM, XGBoost
- Hyperparameter tuning (GridSearchCV / Optuna)
- Feature importance analysis (SHAP values)
- Saving the final pipeline (joblib) and deploying it as an interactive web app

The deployed model is LightGBM with categorical encoding on transaction type.

---

## 2. Dataset
- **Source**: [PaySim](https://www.kaggle.com/ntnu-testimon/paysim1) (synthetic dataset simulating mobile money transactions).  
- **Size**: ~6.3 million transactions  
- **Features used in demo**:
  - amount  
  - oldbalanceOrg  
  - newbalanceOrig  
  - type (OneHot encoded)  

Fraudulent transactions are typically observed when balances don’t match after TRANSFER or CASH_OUT operations.  

---

## 3. Project Workflow
![Workflow](workflow.png)

---

## 4. Exploratory Data Analysis (EDA)
The transaction amounts are highly skewed, with many small transactions and a few very large ones.  
Applying a log transformation makes the distribution more balanced.

![EDA Amount Distribution](eda_amount.png) 
![Log Amount distribution](eda_logamount.png)

---

## 4. Model Training & Evaluation

We tested **Logistic Regression, LightGBM, and XGBoost**.  
LightGBM provided the best trade-off across metrics.

📊 **Performance metrics:**

![Model metrics](model_metrics.png)

| Model  | PR-AUC | ROC-AUC | Recall@0.5% |
|--------|--------|---------|--------------|
| XGBoost| 0.342  | 0.938   | 0.499 |
| LightGBM | **0.344** | **0.933** | **0.495** |
| Logistic Regression | 0.061 | 0.916 | 0.239 |

---

## 5.Feature Importance
We report LightGBM’s built-in feature importances (gain).  
The most influential features are transaction amount, balances, and transaction type.

![Feature Importance](feature_importance.png)

---

## 6. Deployment – HuggingFace Space

The trained pipeline was deployed with **Gradio** on Hugging Face Spaces.  
Users can interactively test the model by entering transaction details and adjusting the **decision threshold**.

📷 **Demo screenshot:**

**![HF Demo](hf_demo.png)**

---

## 7. How to Use

1. Enter transaction details (`amount`, `old balance`, `new balance`, `type`).  
2. Adjust the **decision threshold** (default = 0.5).  
3. Click **Submit**.  
4. The app returns:
   - Fraud probability (0–1)  
   - Decision label (FRAUD / NOT FRAUD)  
   - Model version  

---

## 8. Next Steps

- Extend pipeline with additional features (`dest balances`, engineered features).  
- Improve handling of extreme class imbalance.  
- Deploy REST API version (FastAPI + Docker).  
- Add monitoring dashboards for real-world use cases.  

---

## 9. License

This demo is for **educational and portfolio purposes only**.  
Dataset credit: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)