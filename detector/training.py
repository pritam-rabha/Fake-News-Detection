from pathlib import Path
import hashlib
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from .data_loader import load_dataset
from .preprocessing import preprocess_text
from .models import MODEL_NAMES, build_pipeline, tune_model

def evaluate(y_true, y_pred):
    labels = ["fake", "real"]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }

def train_all(dataset_path, output_dir, tune=True):
    df = load_dataset(dataset_path)
    X, y = df["text"].map(preprocess_text), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    results, best_name, best_score = {}, None, -1
    for name in MODEL_NAMES:
        if tune:
            search = tune_model(name, X_train, y_train)
            model, params = search.best_estimator_, search.best_params_
        else:
            model, params = build_pipeline(name), {}
            model.fit(X_train, y_train)
        metrics = evaluate(y_test, model.predict(X_test))
        metrics["best_params"] = params
        metrics["test_samples"] = int(len(y_test))
        results[name] = metrics
        joblib.dump(model, out / (name.lower().replace(" ", "_") + ".joblib"))
        if metrics["f1_macro"] > best_score:
            best_name, best_score = name, metrics["f1_macro"]
    final_model = build_pipeline(best_name)
    final_model.fit(X, y)
    joblib.dump(final_model, out / "best_model.joblib")
    dataset_digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    summary = {"selected_model": best_name, "dataset_rows": int(len(df)), "dataset_signature": dataset_digest, "test_size": 0.20, "test_samples": int(len(y_test)), "label_counts": {str(k): int(v) for k,v in y.value_counts().to_dict().items()}, "metrics": results, "note": "Metrics are from a hold-out split and do not establish real-world fact-checking accuracy."}
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
