from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

MODEL_NAMES = [
    "Logistic Regression",
    "Multinomial Naive Bayes",
    "Linear SVM",
    "Random Forest",
]
def build_pipeline(model_name):
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=False, max_features=25000, ngram_range=(1, 2))), ("classifier", estimators[model_name])])

def tune_model(model_name, X_train, y_train):
    params = {"tfidf__max_features": [10000, 25000], "tfidf__ngram_range": [(1, 1), (1, 2)]}
    if model_name in ("Logistic Regression", "Linear SVM"):
        params["classifier__C"] = [0.5, 1.0, 2.0]
    elif model_name == "Multinomial Naive Bayes":
        params["classifier__alpha"] = [0.5, 1.0]
    else:
        params["classifier__max_depth"] = [None, 30]
    folds = max(2, min(3, int(y_train.value_counts().min())))
    search = GridSearchCV(build_pipeline(model_name), params, cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42), scoring="f1_macro", n_jobs=-1)
    return search.fit(X_train, y_train)
