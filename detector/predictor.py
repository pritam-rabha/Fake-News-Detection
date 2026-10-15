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
            raise ValueError(
                f"Unsupported model: {self.model_name}"
            )

        self.model_path = (
            self.model_dir / MODEL_FILES[self.model_name]
        )

        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        """Load the selected model, training models if necessary."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            train_all(
                self.dataset_path,
                self.model_dir,
                tune=False,
            )

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

    @staticmethod
    def _contains_any(text, terms):
        """Return True if any supplied phrase appears in the text."""
        return any(term in text for term in terms)

    def _check_suspicious_patterns(self, text):
        """
        Screen for selected suspicious message patterns.

        This is a simple heuristic, not a fact-checking system.
        A matching pattern does not prove that a story is false.
        """
        raw_text = " ".join(str(text).lower().split())

        scholarship_terms = [
            "scholarship",
            "student benefit",
            "education benefit",
            "student grant",
            "college student",
        ]

        payment_terms = [
            "verification fee",
            "processing fee",
            "registration fee",
            "pay ₹",
            "pay rs",
            "payment required",
        ]

        sensitive_terms = [
            "bank otp",
            "bank details",
            "share your otp",
            "provide your otp",
            "send your otp",
        ]

        urgency_terms = [
            "expires tonight",
            "within 30 minutes",
            "immediately",
            "urgent",
            "today only",
        ]

        currency_terms = [
            "₹500 notes",
            "500 notes",
            "500 rupee notes",
        ]

        invalidation_terms = [
            "become invalid",
            "will be invalid",
            "invalid tonight",
            "exchange their notes",
        ]

        link_terms = [
            "clicking the link",
            "click this link",
            "unknown link",
            "click here",
        ]

        prize_terms = [
            "won a free iphone",
            "free iphone",
            "won a prize",
            "government giveaway",
        ]

        has_scholarship = self._contains_any(
            raw_text, scholarship_terms
        )

        has_payment = self._contains_any(
            raw_text, payment_terms
        )

        has_sensitive_request = self._contains_any(
            raw_text, sensitive_terms
        )

        has_urgency = self._contains_any(
            raw_text, urgency_terms
        )

        scholarship_scam = (
            has_scholarship
            and (has_payment or has_sensitive_request)
            and has_urgency
        )

        currency_panic = (
            self._contains_any(raw_text, currency_terms)
            and self._contains_any(raw_text, invalidation_terms)
            and (
                has_urgency
                or self._contains_any(raw_text, link_terms)
            )
        )

        prize_scam = (
            self._contains_any(raw_text, prize_terms)
            and (has_payment or has_sensitive_request)
        )

        return (
            scholarship_scam
            or currency_panic
            or prize_scam
        )

    def predict(self, text):
        """Predict a news label and return the result for the UI."""
        cleaned = preprocess_text(text)

        if not cleaned.strip():
            raise ValueError(
                "The text contains no usable words after preprocessing."
            )

        # Obtain the machine-learning prediction.
        model_label = str(self.model.predict([cleaned])[0])

        # Apply the additional suspicious-pattern screening.
        rule_override = self._check_suspicious_patterns(text)

        label = "fake" if rule_override else model_label

        # Do not report model confidence as though it were the
        # confidence of a rule-based override.
        confidence = None

        if (
            not rule_override
            and hasattr(self.model, "predict_proba")
        ):
            probabilities = self.model.predict_proba([cleaned])[0]
            class_labels = list(map(str, self.model.classes_))

            confidence = float(
                probabilities[class_labels.index(model_label)]
            )

        return {
            "label": label,
            "label_display": (
                "Likely Real"
                if label == "real"
                else "Likely Fake"
            ),
            "confidence": (
                round(confidence * 100, 1)
                if confidence is not None
                else None
            ),
            "model": self.model_name,
            "caveat": (
                "This result combines machine-learning predictions "
                "with suspicious-pattern screening. It is not a "
                "verified fact-check. Verify important claims using "
                "reliable independent sources."
            ),
        }