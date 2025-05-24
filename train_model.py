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
CSV_FILE = "test.csv"

# Load CSV
df = pd.read_csv(CSV_FILE)

# Preprocess
label_map = {1: "Very Bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Very Good"}
df["class_index"] = df["class_index"].map(label_map)
df["text"] = df.apply(lambda x: " ".join(filter(None, [str(x["review_title"]), str(x["review_text"])])), axis=1)
df = df[df["text"].str.strip().astype(bool)]
df["malicious"] = df["class_index"].isin(["Very Bad", "Bad"]).astype(int)
df["unworthy"] = df["class_index"].isin(["Neutral"]).astype(int)

# Stratified sampling (again, optional)
df["stratify_key"] = df["malicious"].astype(str) + df["unworthy"].astype(str)
df_sampled = df.groupby("stratify_key", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 10000), random_state=42)
).reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_sampled["text"], df_sampled[["malicious", "unworthy"]], test_size=0.2, random_state=42
)

# Tokenizer & device
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    problem_type="multi_label_classification"
)
model.to(device)

# Optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

# Training loop
model.train()
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}")
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    print(f"Average loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
preds, labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

preds_bin = (np.array(preds) > 0.5)
labels = np.array(labels)

print("\nAccuracy Scores:")
print("Malicious:", accuracy_score(labels[:, 0], preds_bin[:, 0]))
print("Unworthy :", accuracy_score(labels[:, 1], preds_bin[:, 1]))
print("\nClassification Report:")
print(classification_report(labels, preds_bin, target_names=["malicious", "unworthy"]))

# Save
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
print("\n Model and tokenizer saved to 'bert_model'")
