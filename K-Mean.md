# K-Means Clustering

## 1. What is K-Means?

K-Means is an **unsupervised learning algorithm** that groups data into K clusters.

- You choose K (number of groups)
- The algorithm puts similar points into the same group
- No target column. No answers. Just patterns.

---

## 2. Step-by-Step How It Works

### Step 1: Pick K (Number of Groups)

**Example:** K = 3 → you want 3 clusters.

### Step 2: Drop K Random Points (Centers)

- These are called **centroids**
- They are just temporary "group leaders"
- Think of them like flags placed randomly on a map

### Step 3: Assign Each Data Point to the Nearest Centroid

Each data point asks:
> "Which center is closest to me?"

and joins that group.

### Step 4: Move Each Centroid to the Middle of Its Group

- Now each centroid moves to the average position of the points in its group
- That's why it's called **means**

### Step 5: Repeat Steps 3–4

- Keep re-assigning points
- Keep moving centroids
- Until nothing changes anymore
- Then stop. You're done.

---

## 3. Tiny Visual Example

**Data points** = students standing in a field  
**K = 2 teachers**

```
•   •       ×
   •      ×

• •         ×
```

- **Dots** = students
- **×** = teachers

**Process:**
1. Teachers stand randomly
2. Students go to nearest teacher
3. Teachers move to center of their students
4. Repeat
5. Groups stabilize

> **Congratulations, clustering achieved.**

---

## 4. Simple Example

You have customers:

| Age | Spending |
|-----|----------|
| 22  | 15       |
| 25  | 18       |
| 45  | 70       |
| 50  | 75       |

**K = 2**

**Result:**
- **Cluster 1** → young + low spending
- **Cluster 2** → older + high spending

---

## 5. Pros

- Simple
- Fast
- Works well for spherical clusters
- Easy to implement

---

## 6. Cons (Real Ones)

- You must choose K
- Sensitive to initial centroids
- Sensitive to outliers
- Fails on weird shapes
- Distance-based → scaling matters

> **Always scale your data:**

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

---

## 7. Choosing K (Important)

**Bad K → bad clusters.**

### Common Method: Elbow Method

**Plot:**
- **X axis** → K
- **Y axis** → inertia (sum of squared distances)

> Where the curve bends = good K.

---

## 8. Scikit-learn Code

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [22,15],
    [25,18],
    [45,70],
    [50,75]
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
```

---

## 9. One-Line Definition

> **K-Means is an unsupervised algorithm that groups data into K clusters by minimizing distance to cluster centroids.**

