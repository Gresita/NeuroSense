import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "neurosense_cleaned.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_3class.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
print("Dataset path:", DATA_PATH)
print("Dataset exists:", os.path.exists(DATA_PATH))

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 4-class to 3-class mapping
# 0 = Neutral
# 1,2 = Negative
# 3 = Positive
df["label_3class"] = df["label"].map({
    0: 0,
    1: 1,
    2: 1,
    3: 2
})

metadata_cols = ["subject", "session", "trial", "sample"]
target_cols = ["label", "label_3class", "emotion", "Emotion", "label_name"]

drop_cols = metadata_cols + target_cols

feature_cols = [
    col for col in df.columns
    if col not in drop_cols
]

X = df[feature_cols]

# Keep only numeric features
X = X.select_dtypes(include=["number"])
feature_cols = X.columns.tolist()

y = df["label_3class"]
groups = df["subject"]

print("Features:", len(feature_cols))
print("Feature sample:", feature_cols[:10])
print("Classes:", sorted(y.unique()))

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model...")
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", acc)
print("F1-score:", f1)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Neutral", "Negative", "Positive"]
))

joblib.dump({
    "model": model,
    "feature_cols": feature_cols,
    "label_names": {
        0: "Neutral",
        1: "Negative",
        2: "Positive"
    }
}, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")