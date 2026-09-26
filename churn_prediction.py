import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. Dataset load
# ============================================================

df = pd.read_csv("customer_churn.csv")


# ============================================================
# 2. Basic Data Information
# ============================================================

print("\n===== Dataset Information =====")

print("Dataset Shape:", df.shape)

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. Customer ID remove
# ============================================================

df = df.drop("customer_id", axis=1)


# ============================================================
# 4. Features (X) and Target (y) separate
# ============================================================

X = df.drop("churn", axis=1)
y = df["churn"]


# ============================================================
# 5. Categorical columns identify
# ============================================================

categorical_columns = X.select_dtypes(include="str").columns

print("\n===== Categorical Columns =====")
print(categorical_columns)


# ============================================================
# 6. Categorical data encoding
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


print("\n===== Encoded Data =====")
print(X_encoded.head())

print("\nEncoded Data Types:")
print(X_encoded.dtypes)


# ============================================================
# 7. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)


print("\n===== Train-Test Split =====")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# 8. Logistic Regression
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. Logistic Regression Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


print("\n===== Logistic Regression =====")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1-Score:", f1)

print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 10. Logistic Regression Classification Report
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 11. Decision Tree
# ============================================================

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_prob = tree_model.predict_proba(X_test)[:, 1]


# ============================================================
# 12. Decision Tree Evaluation
# ============================================================

tree_accuracy = accuracy_score(y_test, tree_pred)

tree_precision = precision_score(y_test, tree_pred)

tree_recall = recall_score(y_test, tree_pred)

tree_f1 = f1_score(y_test, tree_pred)

tree_roc_auc = roc_auc_score(y_test, tree_prob)

tree_cm = confusion_matrix(y_test, tree_pred)


print("\n===== Decision Tree =====")

print("Accuracy:", tree_accuracy)

print("Precision:", tree_precision)

print("Recall:", tree_recall)

print("F1-Score:", tree_f1)

print("ROC-AUC:", tree_roc_auc)

print("\nConfusion Matrix:")
print(tree_cm)


# ============================================================
# 13. Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(X_test)[:, 1]


# ============================================================
# 14. Random Forest Evaluation
# ============================================================

rf_accuracy = accuracy_score(y_test, rf_pred)

rf_precision = precision_score(y_test, rf_pred)

rf_recall = recall_score(y_test, rf_pred)

rf_f1 = f1_score(y_test, rf_pred)

rf_roc_auc = roc_auc_score(y_test, rf_prob)

rf_cm = confusion_matrix(y_test, rf_pred)


print("\n===== Random Forest =====")

print("Accuracy:", rf_accuracy)

print("Precision:", rf_precision)

print("Recall:", rf_recall)

print("F1-Score:", rf_f1)

print("ROC-AUC:", rf_roc_auc)

print("\nConfusion Matrix:")
print(rf_cm)


# ============================================================
# 15. Model Comparison
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy,
        tree_accuracy,
        rf_accuracy
    ],

    "Precision": [
        precision,
        tree_precision,
        rf_precision
    ],

    "Recall": [
        recall,
        tree_recall,
        rf_recall
    ],

    "F1-Score": [
        f1,
        tree_f1,
        rf_f1
    ],

    "ROC-AUC": [
        roc_auc,
        tree_roc_auc,
        rf_roc_auc
    ]
})


print("\n===== Model Comparison =====")
print(comparison)


# ============================================================
# 16. Hyperparameter Tuning - Random Forest
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=param_grid,

    scoring="f1",

    cv=5,

    n_jobs=-1
)


grid_search.fit(X_train, y_train)


# ============================================================
# 17. Best Parameters
# ============================================================

print("\n===== Hyperparameter Tuning =====")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(grid_search.best_score_)


# ============================================================
# 18. Tuned Random Forest Evaluation
# ============================================================

tuned_rf = grid_search.best_estimator_

tuned_pred = tuned_rf.predict(X_test)

tuned_prob = tuned_rf.predict_proba(X_test)[:, 1]


tuned_accuracy = accuracy_score(
    y_test,
    tuned_pred
)

tuned_precision = precision_score(
    y_test,
    tuned_pred
)

tuned_recall = recall_score(
    y_test,
    tuned_pred
)

tuned_f1 = f1_score(
    y_test,
    tuned_pred
)

tuned_roc_auc = roc_auc_score(
    y_test,
    tuned_prob
)

tuned_cm = confusion_matrix(
    y_test,
    tuned_pred
)


print("\n===== Tuned Random Forest =====")

print("Accuracy:", tuned_accuracy)

print("Precision:", tuned_precision)

print("Recall:", tuned_recall)

print("F1-Score:", tuned_f1)

print("ROC-AUC:", tuned_roc_auc)

print("\nConfusion Matrix:")
print(tuned_cm)


# ============================================================
# 19. Final Model Selection
# ============================================================
# Based on the current test results, Logistic Regression
# performs better than the tested tree-based models.

final_model = model

final_model_name = "Logistic Regression"


print("\n===== Final Model =====")
print("Selected Model:", final_model_name)


# ============================================================
# 20. Final Model Evaluation
# ============================================================

final_pred = final_model.predict(X_test)

final_prob = final_model.predict_proba(X_test)[:, 1]


final_accuracy = accuracy_score(
    y_test,
    final_pred
)

final_precision = precision_score(
    y_test,
    final_pred
)

final_recall = recall_score(
    y_test,
    final_pred
)

final_f1 = f1_score(
    y_test,
    final_pred
)

final_roc_auc = roc_auc_score(
    y_test,
    final_prob
)

final_cm = confusion_matrix(
    y_test,
    final_pred
)


print("\n===== Final Model Evaluation =====")

print("Accuracy:", final_accuracy)

print("Precision:", final_precision)

print("Recall:", final_recall)

print("F1-Score:", final_f1)

print("ROC-AUC:", final_roc_auc)

print("\nFinal Confusion Matrix:")
print(final_cm)


# ============================================================
# 21. Final Classification Report
# ============================================================

print("\n===== Final Classification Report =====")

print(
    classification_report(
        y_test,
        final_pred
    )
)


# ============================================================
# 22. Logistic Regression Feature Coefficients
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": final_model.coef_[0]
})


feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)


feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)


print("\n===== Feature Importance / Coefficients =====")

print(feature_importance)


# ============================================================
# 23. Feature Importance Visualization
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Coefficient"]
)

plt.xlabel("Coefficient")

plt.ylabel("Feature")

plt.title(
    "Logistic Regression Feature Coefficients"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 24. Churn Distribution Visualization
# ============================================================

plt.figure(figsize=(6, 5))

df["churn"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Churn")

plt.ylabel("Number of Customers")

plt.title("Customer Churn Distribution")

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"],
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 25. Monthly Charges vs Churn
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["monthly_charges"],
    df["churn"],
    alpha=0.5
)

plt.xlabel("Monthly Charges")

plt.ylabel("Churn")

plt.title(
    "Monthly Charges vs Churn"
)

plt.tight_layout()

plt.show()


# ============================================================
# 26. Tenure vs Churn
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["tenure"],
    df["churn"],
    alpha=0.5
)

plt.xlabel("Tenure")

plt.ylabel("Churn")

plt.title(
    "Tenure vs Churn"
)

plt.tight_layout()

plt.show()


# ============================================================
# 27. Save Final Model
# ============================================================

joblib.dump(
    final_model,
    "customer_churn_model.pkl"
)


# Save feature columns as well
joblib.dump(
    X_encoded.columns.tolist(),
    "customer_churn_features.pkl"
)


print("\n===== Model Saving =====")

print(
    "Final model saved as: "
    "customer_churn_model.pkl"
)

print(
    "Feature columns saved as: "
    "customer_churn_features.pkl"
)


# ============================================================
# 28. Load Saved Model
# ============================================================

loaded_model = joblib.load(
    "customer_churn_model.pkl"
)

loaded_features = joblib.load(
    "customer_churn_features.pkl"
)


print("\n===== Saved Model Loaded Successfully =====")


# ============================================================
# 29. New Customer Prediction Function
# ============================================================

def predict_customer_churn(
    gender,
    age,
    tenure,
    monthly_charges,
    contract_type,
    internet_service,
    support_calls,
    payment_method
):

    new_customer = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "tenure": [tenure],
        "monthly_charges": [monthly_charges],
        "contract_type": [contract_type],
        "internet_service": [internet_service],
        "support_calls": [support_calls],
        "payment_method": [payment_method]
    })


    # Encode categorical data
    new_customer_encoded = pd.get_dummies(
        new_customer,
        columns=categorical_columns,
        drop_first=True
    )


    # Make columns exactly the same as training data
    new_customer_encoded = new_customer_encoded.reindex(
        columns=loaded_features,
        fill_value=False
    )


    # Prediction
    prediction = loaded_model.predict(
        new_customer_encoded
    )[0]


    # Probability
    probability = loaded_model.predict_proba(
        new_customer_encoded
    )[0][1]


    if prediction == 1:

        result = "Customer is likely to CHURN"

    else:

        result = "Customer is likely to NOT CHURN"


    return result, probability


# ============================================================
# 30. Test New Customer
# ============================================================

result, probability = predict_customer_churn(
    gender="Male",
    age=35,
    tenure=12,
    monthly_charges=85.50,
    contract_type="Month-to-month",
    internet_service="Fiber optic",
    support_calls=5,
    payment_method="Electronic check"
)


print("\n===== New Customer Prediction =====")

print("Prediction:", result)

print(
    "Churn Probability:",
    probability
)


# ============================================================
# 31. Final Project Status
# ============================================================

print("\n====================================")
print("CUSTOMER CHURN PROJECT COMPLETED")
print("====================================")

print("Final Model:", final_model_name)

print("Accuracy:", final_accuracy)

print("Precision:", final_precision)

print("Recall:", final_recall)

print("F1-Score:", final_f1)

print("ROC-AUC:", final_roc_auc)