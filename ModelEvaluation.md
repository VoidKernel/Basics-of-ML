# Model Evaluation

## 1. Regression Metrics

These are regression evaluation metrics. They tell you how bad your model actually is, in numbers, instead of vibes.

### 1.1 MAE – Mean Absolute Error

$$MAE = \frac{1}{n} \sum |y - \hat{y}|$$

**Meaning:** Average absolute mistake.

### 1.2 MSE – Mean Squared Error

$$MSE = \frac{1}{n} \sum (y - \hat{y})^2$$

**Meaning:** Average of squared mistakes.

### 1.3 RMSE – Root Mean Squared Error

$$RMSE = \sqrt{MSE}$$

**Meaning:** Square root of MSE. Back to original units.

### 1.4 R² – Coefficient of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

**Where:**
- $SS_{res} = \sum (y - \hat{y})^2$
- $SS_{tot} = \sum (y - \bar{y})^2$

**Meaning:** "How much of the variance did the model explain?"

**Range:**
- **1.0** → perfect
- **0** → as good as predicting mean
- **< 0** → model is embarrassing

### Practical Advice (From Someone Who's Seen Things)

- Report RMSE + R² in papers
- Use MAE when explaining to humans
- Optimize MSE during training
- Never trust a model with good R² but terrible RMSE

---

## 2. Classification Metrics

### 2.1 Confusion Matrix (The Source of Everything)

For binary classification:

|                      | **Predicted Positive** | **Predicted Negative** |
|---------------------|------------------------|------------------------|
| **Actual Positive** | TP (True Positive)     | FN (False Negative)    |
| **Actual Negative** | FP (False Positive)    | TN (True Negative)     |

### 2.2 Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Meaning:** "How often the model is correct."

**Problem:** Useless for imbalanced data.

**Example:**  
99% emails are not spam → model predicts "not spam" always → 99% accuracy → zero intelligence.

### 2.3 Precision

$$Precision = \frac{TP}{TP + FP}$$

**Meaning:** "When the model says positive, how often is it right?"

**Important when:** False positives are expensive.

**Examples:**
- Spam filter
- Fraud detection
- Medical tests that scare people

### 2.4 Recall (Sensitivity / TPR)

$$Recall = \frac{TP}{TP + FN}$$

**Meaning:** "Out of all real positives, how many did we catch?"

**Important when:** False negatives are dangerous.

**Examples:**
- Cancer detection
- Intrusion detection
- Missing criminals, terrorists, bugs in production

### 2.5 F1-Score

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Meaning:** Harmonic mean of precision and recall.

**Why:**
- Balances both
- Punishes extreme behavior
- High precision + low recall → bad
- High recall + low precision → also bad

### 2.6 Relationship Summary

| Metric           | Focus                                    |
|------------------|------------------------------------------|
| Accuracy         | Overall correctness                      |
| Precision        | Quality of positive predictions          |
| Recall           | Coverage of real positives               |
| F1-score         | Balance between precision & recall       |
| Confusion Matrix | Everything                               |

### When to Use What

| Situation        | Metric           |
|------------------|------------------|
| Balanced data    | Accuracy         |
| Spam / fraud     | Precision        |
| Medical / safety | Recall           |
| Imbalanced data  | F1-score         |
| Analysis         | Confusion Matrix |

