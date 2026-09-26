# Customer Churn Prediction

## 📌 Project Overview

This is a Machine Learning project that predicts whether a customer is likely to churn or stay.

The project uses customer information such as age, tenure, monthly charges, contract type, internet service, support calls, and payment method.

A Streamlit frontend is also included for making real-time predictions.

---

## 🎯 Problem Statement

The goal of this project is to predict customer churn using Machine Learning.

- 0 = Customer will stay
- 1 = Customer will churn

---

## 📂 Dataset

The dataset contains 500 customer records.

The dataset includes:

- Customer ID
- Gender
- Age
- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Support Calls
- Payment Method
- Churn

The dataset is a synthetic practice dataset.

---

## 🔧 Data Preprocessing

The following steps were performed:

- Checked the dataset
- Checked missing values
- Checked duplicate records
- Removed customer ID from modeling
- Separated features and target
- Applied One-Hot Encoding
- Split data into training and testing sets

Training data: 80%

Testing data: 20%

---

## 🤖 Machine Learning Models

The following classification models were tested:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest hyperparameters were also tuned using GridSearchCV.

The final model used in the application is Logistic Regression.

---

## 📊 Model Performance

The final Logistic Regression model achieved:

- Accuracy: 61%
- Precision: 61%
- Recall: 47%
- F1-Score: 53%
- ROC-AUC: 68%

These results are based on the test dataset used in this project.

---

## 🖥️ Streamlit Frontend

A Streamlit web application was created for real-time customer churn prediction.

The user can enter:

- Gender
- Age
- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Support Calls
- Payment Method

The application then shows:

- Churn Prediction
- Churn Probability

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit
- Git
- GitHub

---

## 📁 Project Structure

Customer_churn/

├── customer_churn.csv  
├── churn_prediction.py  
├── customer_churn_model.pkl  
├── customer_churn_features.pkl  
├── app.py  
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt