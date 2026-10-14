import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("data/news_dataset.csv")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

train_texts = df.loc[train_idx, "text"].tolist()
test_texts = df.loc[test_idx, "text"].tolist()

print("Vectorizing training articles...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print("Finding the closest training article for each test article...")
nn = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)
nn.fit(X_train)

distances, indices = nn.kneighbors(X_test)
similarities = 1 - distances.ravel()

results = pd.DataFrame({
    "test_index": test_idx,
    "nearest_train_index": [train_idx[i] for i in indices.ravel()],
    "similarity": similarities,
})

print("\nNearest-neighbour similarity distribution:")
print(results["similarity"].describe().round(4).to_string())

for threshold in [0.80, 0.90, 0.95, 0.99]:
    count = int((results["similarity"] >= threshold).sum())
    print(
        f"Test articles with similarity >= {threshold:.2f}: "
        f"{count} ({count / len(results) * 100:.2f}%)"
    )

results.to_csv("reports/near_duplicate_candidates.csv", index=False)

print("\nSaved results to reports/near_duplicate_candidates.csv")

print("\nFive highest-similarity pairs:")
for row in results.nlargest(5, "similarity").itertuples():
    train_row = df.loc[row.nearest_train_index]
    test_row = df.loc[row.test_index]

    print(f"\nSimilarity: {row.similarity:.4f}")
    print("Test label:", test_row["label"])
    print("Training label:", train_row["label"])
    print("Test article:", test_row["text"][:250].replace("\n", " "))
    print("Training article:", train_row["text"][:250].replace("\n", " "))
