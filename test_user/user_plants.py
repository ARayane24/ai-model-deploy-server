import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load exported CSV
df = pd.read_csv("test_user/training_data_export.csv")

# Drop non-numeric columns like 'geometry'
if "geometry" in df.columns:
    df = df.drop(columns=["geometry"])

# Optional: check dtypes and drop more non-numeric columns if needed
non_numeric = df.select_dtypes(include=["object", "string"]).columns.tolist()
if non_numeric:
    print("⚠️ Dropping non-numeric columns:", non_numeric)
    df = df.drop(columns=non_numeric)

# Split features and label (change 'class' if your label column is different)
X = df.drop(columns=["class"])
y = df["class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "svm_model.pkl")
print("✅ SVM model saved as 'svm_model.pkl'")
