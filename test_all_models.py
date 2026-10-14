from detector.predictor import NewsDetector
from detector.config import Config
from pathlib import Path

detector = NewsDetector(
    Path(Config.DATASET_PATH),
    Path(Config.MODEL_DIR),
    "Linear SVM",
)

test_cases = [
    {
        "category": "Fabricated government payment",
        "text": (
            "The government has announced that every citizen "
            "will receive 50000 rupees tomorrow. Share this message "
            "with 10 people and register on an unofficial website "
            "to receive the payment."
        ),
    },
    {
        "category": "Fake prize offer",
        "text": (
            "Congratulations! You have won a free iPhone. "
            "Pay a processing fee immediately and send your bank "
            "details to claim your prize."
        ),
    },
    {
        "category": "Ordinary news-style text",
        "text": (
            "The central bank released its annual report on "
            "economic growth, inflation and financial stability."
        ),
    },
    {
        "category": "Unverified political claim",
        "text": (
            "A politician secretly announced a new policy, "
            "but no official statement or reliable report "
            "has confirmed the announcement."
        ),
    },
    {
        "category": "Short ambiguous text",
        "text": "A new announcement has been made today.",
    },
]

models = [
    "Logistic Regression",
    "Multinomial Naive Bayes",
    "Linear SVM",
    "Random Forest",
]

for case in test_cases:
    print("\n" + "=" * 65)
    print("CATEGORY:", case["category"])
    print("TEXT:", case["text"])

    for model_name in models:
        try:
            detector.model_name = model_name
            detector._load_or_train()

            result = detector.predict(case["text"])

            print(
                f'{model_name}: '
                f'{result["label_display"]}'
            )
        except Exception as exc:
            print(f"{model_name}: ERROR - {exc}")
