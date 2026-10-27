 
# Predicting Future Airbnb Review Activity in London

### Machine Learning | Python | XGBoost | Data Analytics

## Project Overview

This project explores whether future Airbnb review activity can be predicted using listing characteristics available at an earlier point in time.

The analysis uses two snapshots of Airbnb listings in London, collected in **September 2025 and June 2026**. The objective is to classify listings based on whether they received **at least five additional reviews** during the observation period.

The project was developed as part of my master's thesis.

## Methodology

The analysis follows an end-to-end machine learning workflow:

1. Data cleaning, feature engineering, and dataset preparation.
2. Train, validation, and test splitting with host-level separation.
3. Comparison of Logistic Regression, Random Forest, XGBoost, and CatBoost.
4. Stratified group cross-validation and XGBoost hyperparameter tuning.
5. Classification threshold optimization on the validation set.
6. Final evaluation on an independent test set.
7. Model interpretation using Permutation Importance and SHAP.

The final model uses **22 predictors**, excluding historical review counts.

## Results

The selected model was a tuned **XGBoost classifier**, with a decision threshold of 0.45.

| Metric | Test score |
|---|---:|
| ROC AUC | 0.8382 |
| PR AUC (Average Precision) | 0.7456 |
| Accuracy | 0.7526 |
| Balanced Accuracy | 0.7593 |
| Precision | 0.6553 |
| Recall | 0.7915 |
| F1-score | 0.7170 |

The model identified approximately **79% of listings** that achieved the target level of future review activity.

Feature importance analysis highlighted availability, booking conditions, and host-related characteristics as important predictive factors.

## Repository Structure

- `01-airbnb-london-data-preparation.ipynb` – Data cleaning, preprocessing, and feature engineering.
- `02-airbnb-london-predictive-modeling.ipynb` – Model training, validation, evaluation, and interpretation.

## Data Source

Data source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), London listings.

The processed dataset is not included in this repository. The notebooks document the analytical workflow and contain saved results, but reproducing the analysis requires access to the corresponding source snapshots and prepared data.

## Tools and Libraries

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, SHAP, Matplotlib, and Jupyter Notebook.

## Limitations

Review activity serves as an indirect indicator of listing activity rather than a direct measure of reservations or revenue. Results are specific to London and the analyzed observation period. Feature importance reflects predictive relationships rather than causation.
