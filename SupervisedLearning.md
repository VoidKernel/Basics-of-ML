# Supervised Learning

## Algorithms Overview

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Neural Networks

---

## 1. Linear Regression

Linear Regression is a machine learning algorithm used to predict a number (a continuous value) by learning a straight-line relationship between inputs and output.

### Simple Formula

$$y = wx + b$$

**Where:**
- $x$ = input (feature)
- $y$ = output (prediction)
- $w$ = weight (slope)
- $b$ = bias (intercept)

### What It Is Used For

- House price prediction
- Sales forecasting
- Salary estimation
- Temperature prediction
- Any task where the answer is a number

---

## 2. Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for **binary classification**.

### How It Works

It starts like Linear Regression:

$$z = wx + b$$

But instead of returning $z$ directly, it passes it through the **sigmoid function**:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Now output is always: $0 \leq \hat{y} \leq 1$

### When to Use

- Binary classification
- Interpretable model needed
- Small to medium datasets
- You want probabilities



