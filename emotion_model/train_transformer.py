import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
train_path = Path("data/processed/train.csv")
val_path = Path("data/processed/val.csv")
model_out = Path("emotion_model/models/transformer_model")
model_out.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Drop NaNs and blanks
train_df = train_df.dropna(subset=["text", "label"])
val_df = val_df.dropna(subset=["text", "label"])
train_df = train_df[train_df["text"].str.strip() != ""]
val_df = val_df[val_df["text"].str.strip() != ""]

# Encode labels
label_encoder = LabelEncoder()
train_df["label_id"] = label_encoder.fit_transform(train_df["label"])
val_df["label_id"] = label_encoder.transform(val_df["label"])

# Save label mapping
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
pd.Series(label_map).to_json(model_out / "label_map.json")

# ----------------------------
# Tokenization
# ----------------------------
print("🔤 Tokenizing text...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# Create datasets and tokenize
train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
val_ds = Dataset.from_pandas(val_df[["text", "label_id"]])
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Rename label column to 'labels' for the model
train_ds = train_ds.rename_column("label_id", "labels")
val_ds = val_ds.rename_column("label_id", "labels")

# Set format with all required columns
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------
# Model
# ----------------------------
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=num_labels
)

# ----------------------------
# Training setup
# ----------------------------
training_args = TrainingArguments(
    output_dir=str(model_out),
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Reduced for CPU training
    per_device_eval_batch_size=8,   # Reduced for CPU training
    warmup_steps=500,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="results/logs",
    logging_steps=100,
    save_total_limit=3,
    save_strategy="epoch",
    report_to="none"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": (preds == labels).mean()
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# ----------------------------
# Train
# ----------------------------
print("🚀 Training DistilBERT model...")
trainer.train()

# ----------------------------
# Save model
# ----------------------------
print("💾 Saving model...")
trainer.save_model(str(model_out))
tokenizer.save_pretrained(str(model_out))
print(f"✅ Model saved to: {model_out}")