# Module 2 – Analytics & Machine Learning Report

## 1. Dataset

The analysis uses the cleaned Titanic dataset stored in:

`analytics/titanic.csv`

The cleaned dataset contains 889 rows and 13 columns.

The target variable for classification is `survived`.

---

## 2. Classification

Three baseline classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The evaluation metrics were:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Class Imbalance Handling

The Titanic target is imbalanced, with fewer surviving passengers than non-survivors.

Two approaches were evaluated:

- Logistic Regression using `class_weight="balanced"`
- Logistic Regression using SMOTE

These approaches were compared with the baseline models.

### Random Forest Tuning

Random Forest hyperparameters were optimized using GridSearchCV.

Best parameters:

- `max_depth = None`
- `min_samples_split = 5`
- `n_estimators = 200`

Best cross-validation F1 score:

`0.7453`

The tuned Random Forest also produced an OOB score of approximately:

`0.8073`

### Model Selection

The Random Forest models provided strong overall classification performance.

The tuned Random Forest was selected as the final classification model and saved as:

`analytics/outputs/best_random_forest_pipeline.joblib`

The generated confusion matrices and ROC curve provide additional evaluation evidence.

---

## 3. Regression

A regression model was developed to predict Titanic passenger fare.

The evaluation metrics were:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Adjusted R²

### Results

Training rows: 711

Testing rows: 178

MAE: 18.3754

RMSE: 41.2921

R²: 0.3691

Adjusted R²: 0.2708

The preprocessing stage produced 22 predictors.

### Interpretation

The regression model explains approximately 36.9% of the variation in passenger fare on the test data.

The difference between R² and adjusted R² indicates that the number of predictors should be considered when evaluating model usefulness.

The residual analysis indicates possible heteroscedasticity, meaning that prediction errors vary across different fare ranges.

The regression pipeline was saved as:

`analytics/outputs/fare_regression_pipeline.joblib`

---

## 4. Output Files

The `analytics/outputs` directory contains:

- EDA charts
- Missing-value report
- Standardization checks
- Classification results
- Confusion matrices
- ROC curves
- Classification summary
- Random Forest tuned pipeline
- Random Forest OOB evaluation
- Regression results
- Regression residual plot
- Fare regression pipeline

---

## 5. Conclusion

The analytics module successfully demonstrates exploratory data analysis, classification, class-imbalance handling, model comparison, hyperparameter tuning, and regression.

For classification, Random Forest provides a strong overall model and was selected as the final tuned model.

For regression, the fare model provides moderate predictive power, with an R² of approximately 0.369. The residual analysis suggests possible heteroscedasticity, which should be considered when interpreting fare predictions.