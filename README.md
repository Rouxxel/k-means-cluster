# Customer Segmentation Using KMeans Clustering

This project applies an **Unsupervised Machine Learning** technique—**KMeans Clustering**—to segment e-commerce customers into distinct groups based on their purchasing behavior.

---

## 🎯 Objective

The goal is to group similar customers together using clustering so that businesses can better understand customer behavior and tailor marketing strategies accordingly.

---

## 📦 Dataset

The dataset contains transactional records from an online retailer, including invoice details, customer IDs, quantity purchased, unit price, and country.

### Key Raw Features:
- `InvoiceNo`: Unique identifier for each transaction
- `CustomerID`: Unique identifier for each customer
- `Quantity`: Number of units purchased
- `UnitPrice`: Price per unit of the product
- `InvoiceDate`: Date of purchase
- `Country`: Country of the customer

---

## 🧮 Features Used for Clustering

After cleaning and preprocessing, the model uses the following customer-level aggregated features:

- `NumPurchases`: Number of unique purchases (invoices)
- `TotalQuantity`: Total number of items purchased
- `TotalSpent`: Total monetary value spent

Additional transformations:
- `LogTotalQuantity`: Log-transformed quantity (to reduce skew)
- `LogTotalSpent`: Log-transformed amount spent (to reduce skew)

---

## 🧪 Steps

1. **Data Cleaning & Preprocessing**
   - Handle missing values
   - Remove outliers based on quantiles
   - Feature engineering (e.g., `TotalSpent`)
   - Aggregate transaction-level data to customer-level

2. **Feature Scaling**
   - Standardize features using `StandardScaler`

3. **Finding Optimal Clusters**
   - Apply the **Elbow Method** by plotting Within-Cluster Sum of Squares (Inertia) vs number of clusters to find the “knee” or “elbow” point which suggests the optimal `k`.
   - Optionally, use the `KneeLocator` algorithm to programmatically identify this point.

4. **Model Training**
   - Use `KMeans` from `scikit-learn` to cluster the scaled features

5. **Evaluation**
   - Evaluate cluster quality using multiple metrics:
     - **Inertia (WSS):** Measures compactness of clusters; lower values indicate tighter clusters.
     - **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters; scores near +1 indicate well-defined clusters, scores near 0 indicate overlapping clusters.
     - **Calinski-Harabasz Index:** Ratio of between-clusters variance to within-cluster variance; higher is better.
     - **Davies-Bouldin Index:** Measures cluster separation; lower values indicate better separation.
   - Silhouette analysis is also used for visual validation and choosing the optimal number of clusters by comparing scores across different `k`.

6. **Visualization**
   - Use **PCA** and **t-SNE** for 2D visualizations of customer segments
   - Visualize cluster quality with **Silhouette plots** to observe cohesion and separation of individual clusters.


7. **Saving Model**
   - Save both the trained KMeans model and the scaler for future use with `joblib`

8. **Prediction Example**
   - Show how to classify a new customer using the trained model

---

## 📊 Example Output: Cluster Profiling

Each cluster is analyzed based on average purchases, total quantity, and total spending:

| Cluster | Avg Purchases | Avg Quantity | Avg Spent | Num Customers |
|---------|---------------|--------------|-----------|---------------|
| 0       | 5.2           | 135.4        | 2450.8    | 142           |
| 1       | 8.3           | 90.3         | 1800.2    | 217           |
| ...     | ...           | ...          | ...       | ...           |

---

## 🧠 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib`
- `os`

---

## 🛠 Future Improvements

- Apply DBSCAN or Hierarchical clustering for comparison
