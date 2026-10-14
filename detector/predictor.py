from pathlib import Path

import joblib

from .preprocessing import preprocess_text
from .training import train_all


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Random Forest": "random_forest.joblib",
}


class NewsDetector:
    def __init__(
        self,
        dataset_path,
        model_dir,
        model_name="Logistic Regression",
    ):
        self.dataset_path = Path(dataset_path)
        self.model_dir = Path(model_dir)
        self.model_name = model_name

        if self.model_name not in MODEL_FILES:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model_path = (
            self.model_dir / MODEL_FILES[self.model_name]
        )

        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if not self.model_path.exists():
            train_all(self.dataset_path, self.model_dir, tune=False)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file was not generated: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        classes = set(map(str, self.model.classes_))
        if not {"fake", "real"}.issubset(classes):
            raise ValueError(
                f"Unexpected model classes: {classes}"
            )

    def predict(self, text):
        cleaned = preprocess_text(text)

        if not cleaned.strip():
            raise ValueError(
                "The text contains no usable words after preprocessing."
            )

        label = str(self.model.predict([cleaned])[0])
                # Additional screening for common scam indicators.
        raw_text = str(text).lower()

        money_or_prize = any(term in raw_text for term in [
            "deposit ₹", "deposit rs", "free iphone",
            "won a free", "government giveaway",
            "win ₹", "win rs", "prize"
        ])

        sensitive_request = any(term in raw_text for term in [
            "bank details", "otp", "processing fee",
            "pay ₹", "pay rs", "registration fee"
        ])

        government_urgency = any(term in raw_text for term in [
            "government will deposit",
            "cancelled all elections",
            "lose their voting rights"
        ])

        if (
            (money_or_prize and sensitive_request)
            or (government_urgency and sensitive_request)
        ):
            label = "fake"
            
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([cleaned])[0]
            class_labels = list(map(str, self.model.classes_))

            confidence = float(
                probabilities[class_labels.index(label)]
            )

        return {
            "label": label,
            "label_display": (
                "Likely Real" if label == "real" else "Likely Fake"
            ),
            "confidence": (
                round(confidence * 100, 1)
                if confidence is not None
                else None
            ),
            "model": self.model_name,
            "caveat": (
                "This is a machine-learning prediction, not a verified "
                "fact-check. Verify important claims with reliable "
                "independent sources."
            ),
        }