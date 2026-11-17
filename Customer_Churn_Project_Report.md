# Customer Churn Prediction Project Report

## 1. Problem Statement
The objective of this project is to predict whether a telecom customer is likely to leave the company (churn). Early identification of potential churn customers helps the business take preventive actions and improve customer retention.

## 2. Dataset Description
The project uses the Telco Customer Churn dataset containing customer information such as:
- Customer demographics and family information.
- Tenure and contract details.
- Internet and subscribed services.
- Monthly charges and total charges.
- Customer churn status.

The data was divided into 80% training data and 20% testing data.

## 3. Methodology
Data preprocessing included handling missing values, numerical scaling and categorical encoding.

The following additional features were created:
- Tenure Group
- Total Services
- Average Spend Per Month
- Has Family
- Is Long Contract

Three machine learning algorithms were developed and compared:
1. Logistic Regression
2. Random Forest
3. XGBoost

RandomizedSearchCV was used to tune Logistic Regression and XGBoost using ROC-AUC as the scoring metric.

Logistic Regression was selected as the final model because it achieved the best cross-validation ROC-AUC score.

A classification threshold of 0.50 was selected based on the threshold comparison, with the objective of balancing precision and recall.

## 4. Model Comparison

| Model | ROC-AUC | F1 Score |
|---|---:|---:|
| Logistic Regression | 0.8459 | 0.6276 |
| XGBoost | 0.8360 | 0.6229 |
| Random Forest | 0.8292 | 0.6070 |

## 5. Final Model Results

The selected Logistic Regression model was evaluated on the held-out test dataset.

- ROC-AUC: 0.8407
- F1 Score: 0.6147
- Accuracy: 74%

**Confusion Matrix:**

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 742 | 293 |
| Actual Churn | 78 | 296 |

The model correctly identified 296 churn customers and missed 78 actual churn customers.

The most influential features based on permutation importance were Internet Service, Tenure, Monthly Charges, Total Services and Long-Term Contract.

## 6. Limitations
- The model may produce false-positive predictions.
- The dataset may not represent all telecom customers.
- The selected threshold is based on model performance rather than a complete financial cost analysis.
- Customer behaviour may change over time, requiring model updates.

## 7. Business Recommendations
- Identify customers with high predicted churn probabilities.
- Offer targeted discounts and retention plans.
- Improve service quality and customer support.
- Encourage customers to choose long-term contracts.
- Monitor customer behaviour and retrain the model periodically.

## 8. Conclusion
The project successfully developed a customer churn prediction system using machine learning. Logistic Regression achieved the best cross-validation ROC-AUC among the evaluated models. The final model was saved using Joblib, and a Python prediction script was developed to generate churn probabilities and customer predictions for new data.