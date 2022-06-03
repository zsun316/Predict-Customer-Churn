# Predict Customer Churn

- A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

## Project Description

- The dataset (https://leaps.analyttica.com/home) consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are 18 features.

## Files and data description
'''bash
LICENSE
README.md
churn_library.py
churn_notebook.ipynb
churn_script_logging_and_tests.py
data
   |-- bank_data.csv
images
   |-- .DS_Store
   |-- eda
   |   |-- churn_distribution.png
   |   |-- customer_age_distribution.png<
   |   |-- heatmap.png
   |   |-- marital_status_distribution.png
   |   |-- total_transaction_distribution.png
   |-- results<
   |   |-- feature_importances.png
   |   |-- logistic_results.png<
   |   |-- rf_results.png
   |   |-- roc_curve_result.png
logs
   |-- churn_library.log
models
   |-- logistic_model.pkl
   |-- rfc_model.pkl
'''

## Requirement
Numpy, Shap, Joblib, Pandas, Sklearn

## Running Files
- There are two ways to test the script:
     ### Directly executing the command (requires main() function):
     $ ipython churn_script_logging_and_tests.py'
     ### Using pytest command:
     $ pytest churn_script_logging_and_tests.py


