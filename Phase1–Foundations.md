# Phase 1 – Foundations

## Core Tools

- **Python** (numpy, pandas)
- **Statistics basics**
- **Scikit-learn**
- **Data preprocessing**

---

## 1. NumPy (Numbers, Fast, No Feelings) 🧠

**Used for:**
- Arrays
- Vectors & matrices
- Linear algebra
- ML math
- Speed

---

## 2. Pandas (Tables, CSVs, Emotional Damage) ✋

**Used for:**
- CSV / Excel
- Cleaning data
- Filtering
- Feature engineering
- Data analysis

---

## 3. Statistics Basics

### 3.1 Types of Data

#### 3.1.1 Numerical (Quantitative)

- **Continuous:** can take any value (height, weight, temperature)
- **Discrete:** countable (number of students, cars)

#### 3.1.2 Categorical (Qualitative)

- **Nominal:** names, labels (red, blue, cat, dog)
- **Ordinal:** ordered categories (small, medium, large; grades A, B, C)

### 3.2 Measures of Central Tendency

These describe "typical" value.

| Measure        | Formula / Idea                    |
|----------------|-----------------------------------|
| Mean (average) | $\frac{\sum x_i}{n}$             |
| Median         | Middle value after sorting        |
| Mode           | Most frequent value               |

**Example:** [1, 2, 2, 3, 4]
- Mean = 2.4
- Median = 2
- Mode = 2

### 3.3 Measures of Spread (Dispersion)

These tell you how scattered your data is.

| Measure                    | Formula / Idea                           |
|----------------------------|------------------------------------------|
| Range                      | Max – Min                                |
| Variance                   | $\frac{\sum (x_i - \bar{x})^2}{n}$      |
| Standard Deviation (SD)    | $\sqrt{Variance}$                        |
| Interquartile Range (IQR)  | Q3 – Q1                                  |

> **High SD** → data all over the place  
> **Low SD** → data hugs the mean

### 3.4 Probability Basics

Probability of event A:

$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total outcomes}}$$

**Example:** roll a die → P(get 3) = 1/6

### 3.5 Common Distributions

- **Normal (Gaussian):** bell curve, mean = median = mode
- **Uniform:** all outcomes equally likely
- **Binomial:** yes/no repeated experiments (coin toss, success/failure)
- **Poisson:** count of events in fixed interval (emails per hour)

> ML loves the normal distribution. Most algorithms assume it somewhere.

### 3.6 Correlation

Measures relationship between two variables:

$$r = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$$

- **r = 1** → perfect positive
- **r = -1** → perfect negative
- **r ≈ 0** → no linear relationship

> Pearson correlation is what ML people usually mean.

### 3.7 Covariance

$$Cov(X, Y) = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{n}$$

- **Positive** → X↑ then Y↑
- **Negative** → X↑ then Y↓
- Magnitude is hard to interpret → use correlation

### 3.8 Skewness & Kurtosis

- **Skewness** → asymmetry of data
- **Kurtosis** → "peakedness" or tail heaviness

> Useful to know if your data is weird before feeding it to ML.

### 3.9 Summary Statistics

```python
import pandas as pd

df = pd.DataFrame({"Score": [10,20,30,40,50]})
df.mean()      # average
df.median()    # middle value
df.std()       # standard deviation
df.var()       # variance
df.describe()  # full summary
```

---

## 4. Scikit-learn

### 4.1 What is Scikit-learn?

- Open-source Python library for ML
- Built on NumPy, SciPy, matplotlib
- Focused on supervised and unsupervised learning
- Provides preprocessing, feature selection, model evaluation, and pipelines

### 4.2 Core Features

| Feature              | What It Does                                                                               |
|---------------------|-------------------------------------------------------------------------------------------|
| Supervised Learning  | Regression, Classification (LinearRegression, LogisticRegression, RandomForest, SVM)      |
| Unsupervised Learning| Clustering, Dimensionality Reduction (KMeans, PCA)                                        |
| Model Evaluation     | Accuracy, Precision, Recall, F1, MSE, R², cross-validation                                |
| Preprocessing        | Scaling, Normalization, Encoding, Imputation                                              |
| Pipelines            | Chain preprocessing + model in one object                                                 |

### 4.3 Basic Workflow in Scikit-learn

1. **Prepare data** → X (features), y (target)
2. **Split data** → train/test
3. **Preprocess** → scale, encode, clean
4. **Train model** → fit()
5. **Predict** → predict()
6. **Evaluate** → metrics

### 4.4 Example: Linear Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data
X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
```

### 4.5 Example: Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
```

### 4.6 Preprocessing Example

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1,100],[2,200],[3,300]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

- **StandardScaler** → mean=0, std=1
- **MinMaxScaler** → scale between 0-1

> ML models love scaled data.

### 4.7 Train/Test Split & Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("CV Scores:", scores)
```

- **Cross-validation** → avoids overfitting
- **cv=5** → 5 folds

### 4.8 Pipelines

Combine preprocessing + model:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

pipeline.fit(X_train, y_train)
```

---

## 5. Scikit-learn ML Workflow (Step by Step)

```
         ┌───────────────┐
         │   Raw Data    │
         └──────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │ Data Cleaning &  │
      │  Preprocessing   │
      │ - Handle NaN     │
      │ - Encode Categorical │
      │ - Feature Scaling │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │ Train/Test Split │
      │  (or CV folds)   │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   Model Choice   │
      │ - Regression     │
      │ - Classification │
      │ - Clustering     │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   Model Training │
      │  model.fit()     │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │   Model Prediction │
      │  model.predict()   │
      └────────┬──────────┘
               │
               ▼
      ┌──────────────────┐
      │ Model Evaluation │
      │ - Regression: MAE, MSE, RMSE, R² │
      │ - Classification: Accuracy, F1, Precision, Recall │
      │ - Cross-validation scores           │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │ Hyperparameter   │
      │  Tuning / Grid   │
      │ - GridSearchCV   │
      │ - RandomizedSearchCV │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │ Final Model      │
      │ - Save using joblib or pickle │
      │ - Deploy / Predict New Data  │
      └──────────────────┘
```
