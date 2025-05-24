
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm.auto import tqdm

# Config
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
USE_FP16 = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
df_fake = pd.read_csv("Fake.csv") 
df_existing = pd.read_csv("test.csv")

# Process fake reviews dataset
df_fake.rename(columns={"text": "text", "label": "fake_label"}, inplace=True)
df_fake["fake_label"] = df_fake["fake_label"].map({"fake": 1, "genuine": 0})
df_fake["malicious"] = 0
df_fake["unworthy"] = 0

# Process existing dataset
label_map = {1: "Very Bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very Good"}
df_existing["class_index"] = df_existing["class_index"].map(label_map)
df_existing["text"] = df_existing.apply(lambda x: " ".join(filter(None, [str(x.get("review_title", "")), str(x.get("review_text", ""))])), axis=1)
df_existing = df_existing[df_existing["text"].str.strip().astype(bool)]
df_existing["malicious"] = df_existing["class_index"].isin(["Very Bad", "Bad"]).astype(int)
df_existing["unworthy"] = df_existing["class_index"].isin(["Neutral"]).astype(int)
df_existing["fake_label"] = 0

# Combine datasets
df_combined = pd.concat([df_existing[["text", "malicious", "unworthy", "fake_label"]],
                         df_fake[["text", "malicious", "unworthy", "fake_label"]]], ignore_index=True)
df_combined.dropna(subset=["text"], inplace=True)
df_combined.drop_duplicates(subset=["text"], inplace=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df_combined["text"],
    df_combined[["malicious", "unworthy", "fake_label"]],
    test_size=0.2,
    stratify=df_combined[["malicious", "unworthy", "fake_label"]],
    random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding="max_length", max_length=MAX_LEN)
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    problem_type="multi_label_classification"
).to(DEVICE)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

# Training
model.train()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(**batch)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    print(f"Average loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model(**batch)
        logits = output.logits
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

preds_bin = (np.array(preds) > 0.5).astype(int)
true_labels = np.array(true_labels)

print("\nAccuracy Scores:")
print("Malicious:", accuracy_score(true_labels[:, 0], preds_bin[:, 0]))
print("Unworthy :", accuracy_score(true_labels[:, 1], preds_bin[:, 1]))
print("Fake     :", accuracy_score(true_labels[:, 2], preds_bin[:, 2]))

print("\nClassification Report:")
print(classification_report(true_labels, preds_bin, target_names=["malicious", "unworthy", "fake"]))

# Save model & tokenizer
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
print("\n✅ Model and tokenizer saved to 'bert_model'")
