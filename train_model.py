import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_text

# Load dataset with custom column names
df = pd.read_csv(
    "dataset/reviews.csv",
    names=["id", "game", "sentiment", "review"]
)

# Keep only required columns
df = df[["review", "sentiment"]]

# Remove missing values
df.dropna(inplace=True)

# Clean review text
df["review"] = df["review"].apply(clean_text)

# Features and labels
X = df["review"]
y = df["sentiment"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer Saved Successfully.")