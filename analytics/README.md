# Module 2 - Analytics Pipeline

## Overview

This module implements an end-to-end analytics and machine learning workflow using the classic Titanic dataset.

The workflow includes data profiling, missing-value handling, exploratory data analysis, visualization, feature preprocessing, classification, imbalance handling, hyperparameter tuning, regression, model evaluation, and saving a complete machine learning pipeline.

The Titanic dataset is loaded once using Seaborn and saved as `titanic.csv` as an offline fallback. The modeling stage uses the same committed CSV instead of loading the dataset again from the internet.

## Project Structure

```text
analytics/
│
├── analytics.ipynb
├── titanic.csv
├── best_titanic_pipeline.joblib
├── README.md
│
└── outputs/
    ├── age_histogram.png
    ├── age_boxplot.png
    ├── fare_histogram.png
    ├── fare_boxplot.png
    ├── correlation_heatmap.png
    ├── survival_sex.png
    ├── survival_pclass.png
    ├── survival_sex_pclass.png
    ├── age_class_survival.png
    ├── age_fare_survival.png
    ├── fare_class_survival.png
    ├── decision_tree.png
    ├── roc_curves.png
    └── residual_plot.png
```

## Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
```

Or use the project's root `requirements.txt`.

## Dataset

The Titanic dataset is loaded using:

```python
sns.load_dataset("titanic")
```

The loaded dataset is immediately saved as:

```text
titanic.csv
```

This file acts as the offline fallback so that the project can be executed even when internet access is unavailable.

The modeling stage reads:

```python
pd.read_csv("titanic.csv")
```

and does not independently call `sns.load_dataset("titanic")`.

## Task 1 — Data Profiling

The dataset was profiled using:

* `df.info()`
* `df.describe()`
* `df.shape`
* Missing-value counts
* Missing-value percentages

The missing-value percentages were calculated for every column containing missing values.

## Task 2 — Missing-Value Handling

The measured missing percentages were:

| Column        | Missing Percentage | Strategy           |
| ------------- | -----------------: | ------------------ |
| `age`         |             19.87% | Median imputation  |
| `embarked`    |              0.22% | Drop affected rows |
| `deck`        |             77.22% | Drop column        |
| `embark_town` |              0.22% | Drop affected rows |

The `age` column had 19.87% missing values, which falls between 5% and 30%, so median imputation was used.

The `embarked` and `embark_town` columns each had approximately 0.22% missing values, which is below the 5% threshold. Therefore, rows containing those missing values were dropped.

The `deck` column had approximately 77.22% missing values. Since this is too high for reliable imputation, the column was dropped rather than introducing potentially misleading values.

## Task 3 — Univariate Analysis

Histograms and box plots were created for:

* Age
* Fare

The IQR method was used to identify outliers.

The outlier limits were calculated using:

```text
Lower limit = Q1 - 1.5 × IQR
Upper limit = Q3 + 1.5 × IQR
```

The exact outlier counts are printed in the notebook.

The fare distribution was identified as right-skewed when the mean was greater than the median and the median was greater than the mode.

The charts are saved in the `outputs` directory.

## Task 4 — Bivariate Analysis

Survival rates were calculated for:

1. Sex
2. Passenger class
3. Sex and passenger class together

The analysis showed that survival was strongly associated with passenger sex and passenger class.

Female passengers generally had higher survival rates than male passengers. First-class passengers generally had better survival outcomes than passengers in lower classes.

A correlation matrix was calculated using exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

The boolean columns `adult_male` and `alone` were excluded because they are derived or redundant variables.

The correlation matrix was visualized using a heatmap.

The two strongest correlations were determined by ranking the absolute values of all off-diagonal correlation coefficients.

The exact values and feature pairs are printed in the notebook.

## Task 5 — Multivariate Data Story

Multiple charts were created to understand the factors associated with survival.

### Survival by Sex

The survival rate differs substantially between male and female passengers. Female passengers generally had a much higher probability of survival.

### Survival by Passenger Class

Passenger class was also strongly related to survival. First-class passengers generally had better survival outcomes than second- and third-class passengers.

### Survival by Sex and Passenger Class

Combining sex and passenger class provides a clearer picture of survival differences. Female passengers, particularly those in higher classes, generally had the strongest survival outcomes.

### Age, Fare and Survival

Age and fare were analyzed together with survival. Higher fares were generally associated with passenger groups having better survival outcomes, while age added another dimension to the survival pattern.

### Fare and Passenger Class

Fare differed considerably between passenger classes. Higher fares were associated with higher passenger classes, connecting economic position with the observed survival differences.

## Task 6 — Standardization Check

The `age` and `fare` columns were standardized using z-score standardization.

The transformed variables were checked to confirm that they had approximately:

```text
Mean = 0
Standard deviation = 1
```

This was an exploratory EDA check only.

The standardized values were not used as input to the final modeling pipeline. The modeling pipeline performs its own training-data-only scaling.

## Task 7 — Train/Test Split

The data was divided into training and testing sets using a stratified split.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification was used to maintain approximately the same survived/not-survived class distribution in both training and testing datasets.

## Task 8 — Preprocessing

The modeling pipeline uses a `ColumnTransformer`.

### Numeric features

The numeric features include:

```text
pclass
age
sibsp
parch
fare
```

Missing numeric values are handled using median imputation, followed by `StandardScaler`.

### Categorical features

The categorical features are:

```text
sex
embarked
```

Missing categorical values are handled using the most frequent value, followed by one-hot encoding.

All preprocessing steps are fitted only on the training data through the scikit-learn pipeline.

The test data is transformed using the already-fitted preprocessing steps.

## Task 9 — Classification Models

Three classification models were trained using the same train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree was visualized using `plot_tree`, including feature names and class names.

## Task 10 — Model Evaluation

All three classifiers were evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC

The results are displayed in the notebook in a comparison table.

The ROC curves are saved as:

```text
outputs/roc_curves.png
```

## Task 11 — Imbalance Handling

The survived/not-survived class balance was examined.

Three approaches were compared:

1. Baseline model
2. `class_weight="balanced"`
3. SMOTE oversampling

SMOTE was applied only to the training fold through the modeling pipeline to prevent test-set leakage.

Precision, recall and F1 score were compared.

The strategy with the highest F1 score was considered the strongest overall imbalance-handling approach because F1 balances precision and recall.

## Task 12 — Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest.

The parameters searched were:

```text
n_estimators
max_depth
max_features
```

The Random Forest was created with:

```python
oob_score=True
```

The notebook reports:

* Best parameter combination
* Best cross-validation score
* Out-of-bag score

The OOB score provides an additional estimate of Random Forest generalization performance.

## Task 13 — Regression Side Task

A multivariate Linear Regression model was created to predict `fare` using the other available features.

The regression model was evaluated using:

* MAE
* RMSE
* R²
* Adjusted R²

A residual plot was also produced.

The residual plot was examined for non-random spread. A systematic increase or decrease in residual spread would indicate heteroscedasticity.

The regression residual plot is saved as:

```text
outputs/residual_plot.png
```

## Task 14 — Final Model Comparison

The classification models are compared using:

```text
Accuracy
Precision
Recall
F1
AUC
```

The regression model is evaluated separately using:

```text
MAE
RMSE
R²
Adjusted R²
```

Classification and regression metrics are not directly comparable because they measure different types of prediction performance. Therefore, the two metric groups are presented separately.

The final classifier recommendation is based on the overall balance of accuracy, precision, recall, F1 and AUC.

The tuned Random Forest is considered for deployment when its evaluation results provide the strongest overall performance.

## Task 15 — Complete Pipeline Saving

The final preprocessing and model are saved together as one complete scikit-learn pipeline.

The saved file is:

```text
best_titanic_pipeline.joblib
```

The pipeline contains:

```text
Preprocessing
    ↓
Imputation
    ↓
Encoding
    ↓
Scaling
    ↓
Final Model
```

The saved pipeline is reloaded using `joblib.load()` and tested using raw input data.

This ensures that the saved artifact can perform preprocessing and prediction end-to-end without requiring manually preprocessed input.

## Output Files

The generated charts are stored in:

```text
outputs/
```

The main outputs are:

```text
age_histogram.png
age_boxplot.png
fare_histogram.png
fare_boxplot.png
correlation_heatmap.png
survival_sex.png
survival_pclass.png
survival_sex_pclass.png
age_class_survival.png
age_fare_survival.png
fare_class_survival.png
decision_tree.png
roc_curves.png
residual_plot.png
```

## Final Recommendation

The final classifier should be selected based on its measured test-set performance rather than accuracy alone. F1 score and AUC are particularly useful because they provide additional information about the balance between precision, recall and overall class discrimination. The model with the strongest overall combination of these metrics is the preferred deployment candidate. The complete preprocessing and model pipeline is saved as `best_titanic_pipeline.joblib` so that new raw passenger records can be processed and predicted consistently.
