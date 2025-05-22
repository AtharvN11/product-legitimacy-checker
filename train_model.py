import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv("test.csv")

# Check columns
required_columns = ["class_index", "review_title", "review_text"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Map numeric class_index to string labels
label_map = {
    1: "Very Bad",
    2: "Bad",
    3: "Neutral",
    4: "Good",
    5: "Very Good"
}
df["class_index"] = df["class_index"].map(label_map)

# Combine review title and text
df["text"] = df.apply(
    lambda x: " ".join(filter(None, [
        str(x["review_title"]) if pd.notna(x["review_title"]) else "",
        str(x["review_text"]) if pd.notna(x["review_text"]) else ""
    ])).strip(),
    axis=1
)

# Filter out rows with empty text
df = df[df["text"].str.strip().astype(bool)]

# Define binary labels
df["malicious"] = df["class_index"].isin(["Very Bad", "Bad"]).astype(int)
df["unworthy"] = df["class_index"].isin(["Neutral"]).astype(int)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df[["malicious", "unworthy"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputClassifier(SVC(probability=True, kernel="linear"))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n🔍 Accuracy Scores:")
print("Malicious:", accuracy_score(y_test["malicious"], y_pred[:, 0]))
print("Unworthy :", accuracy_score(y_test["unworthy"], y_pred[:, 1]))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["malicious", "unworthy"]))

# Save
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")
print("✅ Model and vectorizer saved successfully.")
