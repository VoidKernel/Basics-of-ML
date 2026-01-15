# Data Split Concept

## 1. Why Split Data at All?

Because evaluating a model on the same data it learned from is **cheating**.

It's like:
1. Reading the answers
2. Taking the exam
3. Bragging about your score

So we divide the data into three personalities.

---

## 2. The Three Sets

### ① Training Set (70%)

**Used to:**
- Learn weights
- Fit parameters
- Minimize loss

> This is where the model actually "studies".

### ② Validation Set (15%)

**Used to:**
- Tune hyperparameters (learning rate, depth, k, etc.)
- Choose between models
- Detect overfitting

> The model does NOT learn from this data. It only gets judged.

### ③ Test Set (15%)

**Used to:**
- Final evaluation
- Report performance in papers
- Reality check

> Touched once. At the end. Like a museum artifact.

---

## 3. Visual Flow

```
Raw Dataset (100%)
        |
        |
        v
--------------------------------
| 70% Train | 15% Val | 15% Test |
--------------------------------

Train → fit model
Val   → tune model
Test  → final score
```

---

## 4. What Happens in Practice?

1. Split dataset
2. Train model on training set
3. Evaluate on validation set
4. Adjust hyperparameters
5. Repeat steps 2–4
6. Lock model
7. Evaluate once on test set

> **If you tune using test set → you contaminated it → score becomes a lie.**

---

## 5. Code in Scikit-learn (Correct Way)

Scikit-learn only directly splits into two, so we do it twice.

```python
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Second split: 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
```

**Now:**
- Train = 70%
- Validation = 15%
- Test = 15%

---

## 6. Overfitting Intuition

**If:**
- Training accuracy = 99%
- Validation accuracy = 65%

**Your model memorized instead of learned.**

> Validation set exposes this.

---

## 7. When NOT to Use Validation Set?

**When:**
- Dataset is small
- You use cross-validation

**Then split becomes:**
- Training + Validation via CV
- Separate test set

---

## 8. One-Line Summary

> **Training set teaches the model, validation set tunes the model, test set judges the model.**
