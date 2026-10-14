from pathlib import Path
import hashlib
import json
import pandas as pd

original_path = Path("data/raw/WELFake_Dataset.csv")
current_path = Path("data/news_dataset.csv")
output_path = Path("reports/dataset_audit.json")

if not original_path.exists():
    raise FileNotFoundError(f"Original dataset not found: {original_path}")

if not current_path.exists():
    raise FileNotFoundError(f"Current dataset not found: {current_path}")

original = pd.read_csv(original_path)
current = pd.read_csv(current_path)

required_original = {"title", "text", "label"}
required_current = {"text", "label"}

if not required_original.issubset(original.columns):
    raise ValueError("Original dataset is missing required columns.")

if not required_current.issubset(current.columns):
    raise ValueError("Current dataset is missing required columns.")

def normalize(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())

original = original.dropna(subset=["label"]).copy()
current = current.dropna(subset=["text", "label"]).copy()

original["source_label"] = original["label"].astype(int)
current["label"] = current["label"].astype(str).str.lower().str.strip()

if not set(original["source_label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected source labels; expected 0 and 1.")

if not set(current["label"].unique()).issubset({"fake", "real"}):
    raise ValueError("Unexpected current labels; expected fake and real.")

original["combined_text"] = (
    original["title"].fillna("").astype(str).str.strip()
    + " "
    + original["text"].fillna("").astype(str).str.strip()
).str.strip()

original["normalized"] = original["combined_text"].map(normalize)
current["normalized"] = current["text"].map(normalize)

# Verify how many current articles can be matched to original articles.
original_map = {}

for text, label in zip(original["normalized"], original["source_label"]):
    if text:
        original_map.setdefault(text, set()).add(int(label))

matched = 0
unmatched = 0
label_mismatches = 0
ambiguous_source_labels = 0

for text, label in zip(current["normalized"], current["label"]):
    source_labels = original_map.get(text)

    if not text or not source_labels:
        unmatched += 1
        continue

    matched += 1

    if len(source_labels) > 1:
        ambiguous_source_labels += 1
        continue

    source_label = next(iter(source_labels))
    expected_label = "fake" if source_label == 0 else "real"

    if label != expected_label:
        label_mismatches += 1

report = {
    "original_rows": int(len(original)),
    "current_rows": int(len(current)),
    "original_label_counts": {
        "fake": int((original["source_label"] == 0).sum()),
        "real": int((original["source_label"] == 1).sum()),
    },
    "current_label_counts": {
        str(k): int(v)
        for k, v in current["label"].value_counts().items()
    },
    "current_exact_duplicates": int(
        current["normalized"].duplicated(keep=False).sum()
    ),
    "current_unique_normalized_articles": int(
        current["normalized"].nunique()
    ),
    "matched_to_original": matched,
    "unmatched_current_rows": unmatched,
    "matched_label_mismatches": label_mismatches,
    "matches_with_ambiguous_source_labels": ambiguous_source_labels,
    "original_sha256": hashlib.sha256(
        original_path.read_bytes()
    ).hexdigest(),
    "current_sha256": hashlib.sha256(
        current_path.read_bytes()
    ).hexdigest(),
    "interpretation": (
        "A successful text match checks label conversion, not factual "
        "correctness. Unmatched rows may differ in formatting or content."
    ),
}

output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nAudit saved to: {output_path}")
