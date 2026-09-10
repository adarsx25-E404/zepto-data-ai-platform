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

### Imbalance Comparison Results

| Model Variant | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression Baseline | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Logistic Regression Balanced | 0.7921 | 0.7183 | 0.7500 | 0.7338 | 0.8612 |
| Logistic Regression SMOTE | 0.7978 | 0.7353 | 0.7353 | 0.7353 | 0.8667 |

### Random Forest Tuning

Random Forest hyperparameters were optimized using GridSearchCV.

Best parameters:

- `max_depth = 5`
- `max_features = sqrt`
- `n_estimators = 200`

Best cross-validation F1 score:

`0.7408`

The tuned Random Forest also produced an OOB score of approximately:

`0.8073`

### Model Selection

The tuned Random Forest achieved 0.8315 accuracy, 0.8654 precision, 0.6618 recall, 0.7500 F1 score, and 0.8389 ROC-AUC on the test set.

The tuned Random Forest was selected as the final classification model and saved as:

`analytics/outputs/best_random_forest_pipeline.joblib`

The generated confusion matrices and ROC curve provide additional evaluation evidence.

### Baseline Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7697 | 0.6901 | 0.7206 | 0.7050 | 0.7541 |
| Random Forest | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8196 |

---

---

## 3. Chart Interpretations

### Chart 1 — Survival Rate by Sex

The chart shows a clear difference in survival rates between female and male passengers. Female passengers had a much higher survival rate than male passengers, while the male survival rate was substantially lower. This suggests that sex was an important factor associated with survival in the Titanic dataset.

### Chart 2 — Survival Rate by Passenger Class

The chart shows that survival rates decreased as passenger class increased from first class to third class. First-class passengers had the highest survival rate, while third-class passengers had the lowest. This indicates that passenger class was strongly associated with survival.

### Chart 3 — Survival Rate by Sex and Passenger Class

The chart shows that female passengers generally had higher survival rates than male passengers within each passenger class. Female first- and second-class passengers had particularly high survival rates, while male second- and third-class passengers had much lower survival rates. This shows that sex and passenger class together provide a clearer picture of survival patterns than either variable alone.

### Chart 4 — Fare and Survival Analysis

The fare distribution and survival analysis show that passenger fares were highly uneven, with a smaller number of passengers paying substantially higher fares. Higher passenger classes generally had higher fares and also higher survival rates, suggesting that fare may be related to socioeconomic differences represented by passenger class.





## 4. Regression

A regression model was developed to predict Titanic passenger fare.

The evaluation metrics were:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Adjusted R²

### Results

Training rows: 711

Testing rows: 178

MAE: 18.3735

RMSE: 41.2921

R²: 0.3609

Adjusted R²: 0.2702

The preprocessing stage produced 22 predictors.

### Regression Model Metrics

| Regression Model | MAE | RMSE | R² | Adjusted R² |
|---|---:|---:|---:|---:|
| Multivariate Linear Regression | 18.3735 | 41.2921 | 0.3609 | 0.2702 |

The regression metrics are reported separately from the classification metrics because they measure different types of prediction performance and are not directly comparable on a common numerical scale.

### Interpretation

The regression model explains approximately 36.1% of the variation in passenger fare on the test data.

The difference between R² and adjusted R² indicates that the number of predictors should be considered when evaluating model usefulness.

The residual analysis indicates possible heteroscedasticity, meaning that prediction errors vary across different fare ranges.

The regression pipeline was saved as:

`analytics/outputs/fare_regression_pipeline.joblib`

---

## 5. Output Files

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

## 6. Conclusion

The analytics module successfully demonstrates exploratory data analysis, classification, class-imbalance handling, model comparison, hyperparameter tuning, and regression.

For classification, my preferred model for deployment is the tuned Random Forest because it achieved the highest test accuracy of 0.8315 and the highest precision of 0.8654 among the evaluated classifiers. 
It also achieved an F1 score of 0.7500 and a ROC-AUC of 0.8389, showing strong overall classification performance. 
Although its recall of 0.6618 was lower than the recall of the balanced and SMOTE Logistic Regression models, the tuned Random Forest provided the strongest overall combination of test-set metrics.

For regression, the fare model provides moderate predictive power, with an R² of approximately 0.361. The residual analysis suggests possible heteroscedasticity, which should be considered when interpreting fare predictions.