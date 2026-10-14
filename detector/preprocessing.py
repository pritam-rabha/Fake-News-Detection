import re
from functools import lru_cache

from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

# Preserve negation because it can change the meaning of a claim.
_NEGATION_WORDS = {
    "no", "nor", "not", "never", "neither",
    "nobody", "nothing", "nowhere", "cannot"
}

_STOPWORDS = set(ENGLISH_STOP_WORDS) - _NEGATION_WORDS

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


@lru_cache(maxsize=1)
def _wordnet_available():
    try:
        _LEMMATIZER.lemmatize("articles")
        return True
    except LookupError:
        return False


def preprocess_text(text: str) -> str:
    """Normalize text while preserving important negation words."""
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    result = []

    for token in tokens:
        if token in _STOPWORDS or len(token) < 2:
            continue

        if _wordnet_available():
            token = _LEMMATIZER.lemmatize(token)

        token = _STEMMER.stem(token)

        if token:
            result.append(token)

    return " ".join(result)
