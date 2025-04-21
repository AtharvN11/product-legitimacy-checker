import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer

# Load your dataset
try:
    df = pd.read_csv("test.csv")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Data validation
required_columns = ["class_index", "review_title", "review_text"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Combine review title and text with better null handling
df["text"] = df.apply(
    lambda x: " ".join(filter(None, [
        str(x["review_title"]) if pd.notna(x["review_title"]) else "",
        str(x["review_text"]) if pd.notna(x["review_text"]) else ""
    ])).strip(),
    axis=1
)

# Validate class labels
valid_classes = ["Very Bad", "Bad", "Neutral"]
invalid_classes = df[~df["class_index"].isin(valid_classes)]["class_index"].unique()
if len(invalid_classes) > 0:
    print(f"Warning: Found invalid classes: {invalid_classes}")

# Features and targets with validation
X = df["text"]
if X.isna().any():
    print("Warning: Found null values in text features")
    X = X.fillna("")

# Define binary labels
df["malicious"] = df["class_index"].isin(["Very Bad", "Bad"]).astype(int)
df["unworthy"] = df["class_index"].isin(["Neutral"]).astype(int)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Vectorize text
X_vectorized = vectorizer.fit_transform(X)

# Multi-output model with SVM (you can try LinearSVC or other)
model = MultiOutputClassifier(SVC(probability=True, kernel="linear"))

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, df[["malicious", "unworthy"]], test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save with error handling
try:
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "model.pkl")
    print("✅ Model and vectorizer saved successfully.")
except Exception as e:
    print(f"Error saving model files: {e}")
    raise
