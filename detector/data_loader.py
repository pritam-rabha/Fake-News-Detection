from pathlib import Path
import pandas as pd
REQUIRED_COLUMNS = {"text", "label"}
def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns text,label; missing: {', '.join(sorted(missing))}")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & df["label"].isin(["real", "fake"])]
    if len(df) < 8 or df["label"].nunique() != 2:
        raise ValueError("Dataset needs at least 8 rows and both labels: real and fake.")
    return df.reset_index(drop=True)
