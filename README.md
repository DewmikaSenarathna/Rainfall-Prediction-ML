# 🌧️ Rainfall Prediction Using Machine Learning  

## 📌 Project Overview

This project focuses on the design and implementation of a machine learning-based rainfall prediction system using historical weather data and supervised learning techniques.

Multiple machine learning models are trained, evaluated and compared to determine the most accurate and reliable predictor.

---
<br>

## 🎯 Project Objectives

- Predict rainfall occurrence using machine learning classification  
- Train and evaluate multiple ML models  
- Compare model performance using standard evaluation metrics  
- Provide a reproducible ML pipeline  

---
<br>

## 🧠 Key Technical Concepts

- Supervised machine learning  
- Feature engineering  
- Binary classification  
- Model evaluation  

---
<br>

## 🧩 System Architecture (High-Level)

Inputs:
- Historical weather features  

Processing:
- Data preprocessing  
- Model training  
- Model evaluation  

Outputs:
- Rainfall prediction  
- Performance metrics  

---
<br>

## 📂 Repository Structure

```text
Rainfall_Prediction_ML/
│
├── data/
│   └── Rainfall.csv
│
├── notebooks/
│   └── Rainfall_Prediction_ML.ipynb
│
├── models/
│   ├── DecisionTreeClassifier_model.pkl
│   ├── GaussianNB_model.pkl
│   └── KNeighborsClassifier_model.pkl
│   └── LogisticRegression_model.pkl
│   └── RandomForestClassifier_model.pkl
│   └── SVC_model.pkl
│   └── XGBClassifier_model.pkl
│
├── reports/
│   ├── classification_reports.txt
│   ├── roc_auc_scores.txt
│
├── figures/
│   └── DecisionTreeClassifier_confusion_matrix
│   └── GaussianNB_confusion_matrix
│   └── KNeighborsClassifier_confusion_matrix
│   └── LogisticRegression_confusion_matrix
│   └── RandomForestClassifier_confusion_matrix
│   └── SVC_confusion_matrix
│   └── XGBClassifier_confusion_matrix
│
└── README.md
```
---
<br>

## 🛠 Software and Tools

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Joblib  

---
<br>

## 🔬 Machine Learning Approach

Historical rainfall dataset is collected and preprocessed  

Features are normalized and cleaned  

Dataset is split into training and validation sets  

Multiple ML models are trained:  

- Logistic Regression  
- Support Vector Machine  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- K-Nearest Neighbors  
- XGBoost  

Models are evaluated using:  

- Confusion Matrix  
- Classification Report  
- ROC-AUC Score  

---
<br>
