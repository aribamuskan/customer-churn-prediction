# 👥 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn or stay based on customer demographics, service usage, account information, and support-related features.

The project demonstrates an end-to-end Machine Learning workflow, including data understanding, preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, model persistence, and deployment with Streamlit.

---

## 🚀 Live Demo

Try the deployed application:

**[Customer Churn Prediction · Streamlit](https://customer-churn-prediction-fqzgrhqshddkurxo7eeybf.streamlit.app/)**

---

## 📌 Project Overview

Customer churn prediction is a **binary classification** problem.

The objective of this project is to predict whether a customer is likely to leave a company based on their available customer information.

The model uses features such as:

- Gender
- Age
- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Support Calls
- Payment Method

### Target Variable

```text
0 → Customer will stay
1 → Customer will churn
```

---

## 🎯 Problem Statement

Customer churn is an important business problem because losing existing customers can affect revenue and long-term growth.

The goal of this project is to develop a Machine Learning classification model that can analyze customer information and predict the likelihood of churn.

The trained model is integrated into a Streamlit web application so users can enter customer information and receive a real-time prediction.

---

## 📊 Dataset

The dataset contains **500 customer records**.

The dataset is **synthetic** and was created for Machine Learning practice and portfolio development.

### Dataset Features

| Feature | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `gender` | Customer gender |
| `age` | Customer age |
| `tenure` | Length of time the customer has been with the company |
| `monthly_charges` | Customer's monthly charges |
| `contract_type` | Type of customer contract |
| `internet_service` | Type of internet service |
| `support_calls` | Number of customer support calls |
| `payment_method` | Customer payment method |
| `churn` | Target variable |

### Target Variable

```text
0 = Customer will stay
1 = Customer will churn
```

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
One-Hot Encoding
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Final Model Selection
     ↓
Model Saving
     ↓
Streamlit Deployment
```

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

- Inspected the dataset
- Checked dataset shape
- Checked column names
- Checked data types
- Checked for missing values
- Checked for duplicate records
- Removed `customer_id` from the modeling features
- Separated features and target variable
- Applied One-Hot Encoding to categorical features
- Split the dataset into training and testing sets

### Train/Test Split

```text
Training Data → 80%
Testing Data  → 20%
```

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting customer churn probabilities.

### 2. Decision Tree

Decision Tree was used to learn decision rules from customer features and classify customers into churn and non-churn classes.

### 3. Random Forest

Random Forest was used as an ensemble classification model consisting of multiple decision trees.

---

## ⚙️ Hyperparameter Tuning

Random Forest hyperparameters were tuned using:

```text
GridSearchCV
```

Hyperparameter tuning was performed to evaluate different parameter combinations and improve the model's performance.

---

## 📈 Model Evaluation

The classification models were evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

Using multiple evaluation metrics provides a more complete understanding of classification performance instead of relying only on accuracy.

---

## 📊 Final Model Performance

The final Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 61% |
| Precision | 61% |
| Recall | 47% |
| F1-Score | 53% |
| ROC-AUC | 68% |

These results are based on the test dataset used in this project.

---

## 🏆 Final Model

After comparing the trained classification models, **Logistic Regression** was selected as the final model used in the Streamlit application.

The trained model was saved using `joblib` so that it could be loaded later without retraining.

---

## 💾 Saved Model Files

The project contains the following saved files:

```text
customer_churn_model.pkl
customer_churn_features.pkl
```

### `customer_churn_model.pkl`

Contains the trained Logistic Regression model.

### `customer_churn_features.pkl`

Contains the feature information required to prepare new customer input in the correct format before making predictions.

---

## 🔮 Prediction Workflow

The deployed application follows this process:

```text
User Input
    ↓
Input Validation
    ↓
Categorical Encoding
    ↓
Feature Alignment
    ↓
Saved Machine Learning Model
    ↓
Prediction Probability
    ↓
Churn Prediction
```

The application provides:

- Churn prediction
- Churn probability

---

## 🖥️ Streamlit Web Application

A Streamlit web application was developed to make the Machine Learning model interactive and accessible through a browser.

The user can enter:

- Gender
- Age
- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Support Calls
- Payment Method

The application then generates:

- **Churn Prediction**
- **Churn Probability**

---

## 🌐 Live Application

The trained model has been deployed using Streamlit.

### Live Demo

**[Open Customer Churn Prediction App](https://customer-churn-prediction-fqzgrhqshddkurxo7eeybf.streamlit.app/)**

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Visualization

- Matplotlib

### Deployment

- Streamlit

### Version Control

- Git
- GitHub

---

## 📚 Machine Learning Concepts Practiced

This project provided practical experience with:

- Binary Classification
- Supervised Learning
- Data Understanding
- Data Preprocessing
- Feature Engineering
- Categorical Variables
- One-Hot Encoding
- Train/Test Split
- Logistic Regression
- Decision Tree
- Random Forest
- Model Comparison
- Hyperparameter Tuning
- GridSearchCV
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Probability Prediction
- Model Persistence
- Streamlit Deployment
- Git
- GitHub

---

## 📁 Project Structure

```text
Customer_churn/
│
├── customer_churn.csv
├── churn_prediction.py
├── customer_churn_model.pkl
├── customer_churn_features.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/aribamuskan/customer-churn-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd customer-churn-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Key Project Highlights

- Built a complete customer churn classification pipeline
- Compared multiple Machine Learning algorithms
- Applied categorical feature encoding
- Performed Random Forest hyperparameter tuning using GridSearchCV
- Evaluated models using multiple classification metrics
- Saved the trained Machine Learning model using Joblib
- Built an interactive Streamlit frontend
- Deployed the application online
- Documented the complete project workflow

---

## ⚠️ Disclaimer

This project uses a **synthetic dataset** created for educational, learning, and portfolio practice purposes.

The model is not intended to be used as a production customer retention system without additional validation, real-world data, monitoring, security considerations, and domain-specific testing.

---

## 👩‍💻 Author

**Ariba Muskan**

Software Engineering Student | Machine Learning Enthusiast

---

## ⭐ Project Summary

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction, starting from a raw dataset and progressing through preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, model persistence, and deployment.

The final system allows users to enter customer information through a Streamlit interface and receive a real-time churn prediction and probability.