# Predict Customer Churn

- A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

## Project Description

- The dataset (https://leaps.analyttica.com/home) consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are 18 features.

## Files and data description
>LICENSE<br />
>README.md<br />
>churn_library.py<br />
>churn_notebook.ipynb<br />
>churn_script_logging_and_tests.py<br />
>data<br />
>   |-- bank_data.csv<br />
>images<br />
>   |-- .DS_Store<br />
>   |-- eda<br />
>   |   |-- churn_distribution.png<br />
>   |   |-- customer_age_distribution.png<br />
>   |   |-- heatmap.png<br />
>   |   |-- marital_status_distribution.png<br />
>   |   |-- total_transaction_distribution.png<br />
>   |-- results<br />
>   |   |-- feature_importances.png<br />
>   |   |-- logistic_results.png<br />
>   |   |-- rf_results.png<br />
>   |   |-- roc_curve_result.png<br />
>logs<br />
>   |-- churn_library.log<br />
>models<br />
>   |-- logistic_model.pkl<br />
>   |-- rfc_model.pkl<br />

## Requirement
Numpy, Shap, Joblib, Pandas, Sklearn

## Running Files
- There are two ways to test the script:
     ### Directly executing the command (requires main() function):
     $ ipython churn_script_logging_and_tests.py'
     ### Using pytest command:
     $ pytest churn_script_logging_and_tests.py


