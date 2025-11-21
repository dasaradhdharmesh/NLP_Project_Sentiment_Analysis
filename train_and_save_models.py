# train_and_save_models.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# CONFIG
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Adjust dataset path/columns if different
CSV_PATH = "dataset/Cleaned_Reviews.csv"   # <- change if your file name differs
TEXT_COL = "Cleaned_Review"                # <- change if your text column is different
LABEL_COL = "Sentiment"                    # <- change if your label column is different

print("Loading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.title()

X = df[TEXT_COL].astype(str).tolist()
y = df[LABEL_COL].tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Candidate models
models = {
    "logistic_regression": LogisticRegression(max_iter=400, solver="liblinear"),
    "svm_rbf": SVC(kernel="rbf", probability=True),
    "svm_linear": SVC(kernel="linear", probability=True),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

for key, clf in models.items():
    print(f"\nTraining {key} ...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), stop_words="english")),
        ("clf", clf)
    ])
    pipeline.fit(X_train, y_train)
    ypred = pipeline.predict(X_val)
    print("Val acc:", accuracy_score(y_val, ypred))
    print(classification_report(y_val, ypred, digits=3))
    fname = os.path.join(MODEL_DIR, f"{key}.pkl")
    joblib.dump(pipeline, fname)
    print("Saved ->", fname)

print("All done.")
