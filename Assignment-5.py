import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Task 1: Data Understanding
# ============================================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("="*60)
print("FIRST FIVE RECORDS")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("NUMERICAL FEATURES")
print("="*60)
print(df.select_dtypes(include=["int64","float64"]).columns.tolist())

print("\n" + "="*60)
print("CATEGORICAL FEATURES")
print("="*60)
print(df.select_dtypes(include=["object", "string"]).columns.tolist())

print("\nTarget Variable : Attrition")

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
df.info()

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(df.describe())

# ============================================================
# Task 2: Data Preprocessing
# ============================================================

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

# Remove unnecessary columns if present
if "EmployeeCount" in df.columns:
    df.drop("EmployeeCount", axis=1, inplace=True)

if "Over18" in df.columns:
    df.drop("Over18", axis=1, inplace=True)

if "StandardHours" in df.columns:
    df.drop("StandardHours", axis=1, inplace=True)

if "EmployeeNumber" in df.columns:
    df.drop("EmployeeNumber", axis=1, inplace=True)

# Encode all categorical/string columns
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Task 3: Model Development
# ============================================================

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ============================================================
# Task 4: Model Evaluation
# ============================================================

print("\n" + "="*60)
print("DECISION TREE RESULTS")
print("="*60)

print("Accuracy :", round(accuracy_score(y_test, dt_pred),4))
print("Precision:", round(precision_score(y_test, dt_pred),4))
print("Recall   :", round(recall_score(y_test, dt_pred),4))
print("F1 Score :", round(f1_score(y_test, dt_pred),4))

cm1 = confusion_matrix(y_test, dt_pred)

print("\nConfusion Matrix")
print(cm1)

ConfusionMatrixDisplay(cm1).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)

print("Accuracy :", round(accuracy_score(y_test, rf_pred),4))
print("Precision:", round(precision_score(y_test, rf_pred),4))
print("Recall   :", round(recall_score(y_test, rf_pred),4))
print("F1 Score :", round(f1_score(y_test, rf_pred),4))

cm2 = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix")
print(cm2)

ConfusionMatrixDisplay(cm2).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ============================================================
# Observations
# ============================================================

print("\n" + "="*60)
print("OBSERVATIONS")
print("="*60)

print("1. Random Forest generally achieved higher accuracy than Decision Tree.")
print("2. Random Forest reduced overfitting by combining multiple decision trees.")
print("3. Feature Importance shows which employee attributes influence attrition the most.")
print("4. Random Forest produced more reliable predictions.")

# ============================================================
# Task 5: Conclusion
# ============================================================

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

print("""
This project developed Decision Tree and Random Forest
classification models to predict employee attrition using the
IBM HR Analytics dataset. The dataset was preprocessed by
checking missing values, removing unnecessary columns, and
encoding categorical variables. Both models were trained and
evaluated using Accuracy, Precision, Recall, and F1 Score.
Random Forest generally performed better because it combines
multiple decision trees, reducing overfitting and improving
prediction accuracy. Decision Trees are easy to understand but
can easily overfit the training data. Random Forest is more
accurate but requires more computational resources and is less
interpretable than a single Decision Tree. Overall, Random
Forest is a better choice for employee attrition prediction.
""")