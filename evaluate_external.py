import joblib
import pandas as pd
from pathlib import Path
from detector.preprocessing import preprocess_text

DATA_PATH = Path("data/evaluation/external_test.csv")
MODEL_DIR = Path("artifacts")
OUTPUT_PATH = Path("data/evaluation/predictions.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Random Forest": "random_forest.joblib",
}

df = pd.read_csv(DATA_PATH)

required = {"text", "label", "category"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {sorted(required)}")

results = []

for _, row in df.iterrows():
    text = str(row["text"])
    processed_text = preprocess_text(text)

    print(f"\nCategory: {row['category']}")
    print(f"Text: {text[:150]}")

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"{model_name}: MODEL FILE MISSING")
            continue

        model = joblib.load(model_path)
        prediction = str(model.predict([processed_text])[0])

        result = {
            "category": row["category"],
            "text": text,
            "provisional_label": row["label"],
            "model": model_name,
            "prediction": prediction,
        }
        results.append(result)

        print(f"{model_name}: {prediction}")

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 55)
print("Exploratory evaluation completed.")
print("Predictions saved to:", OUTPUT_PATH)
print("Total predictions:", len(results))
print(
    "\nNote: These examples have not been independently verified."
    "\nDo not interpret provisional labels as verified ground truth."
)
