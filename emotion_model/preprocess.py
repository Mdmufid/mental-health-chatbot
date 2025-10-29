import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """Lowercase, remove URLs, punctuation, and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_goemotions_tsv(path):
    """Load and clean a GoEmotions .tsv file."""
    # GoEmotions format: text\temotion_label(s)\tsentiment
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "sentiment"])
    # In some rows, labels can be comma-separated, we’ll take the first label for simplicity
    df["label"] = df["labels"].astype(str).apply(lambda x: x.split(",")[0])
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]]

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Loading GoEmotions files...")
    train_df = load_goemotions_tsv(raw_dir / "train.tsv")
    val_df = load_goemotions_tsv(raw_dir / "dev.tsv")
    test_df = load_goemotions_tsv(raw_dir / "test.tsv")

    # Save as CSVs
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("✅ Preprocessing complete!")
    print(f"Saved cleaned files to: {out_dir}")
