#@title Fine-Tuning

from transformers import get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config.vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AG News dataset
dataset = load_dataset("ag_news")

def preprocess(batch):
    enc = tokenizer(batch["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=128)
    return {
        "input_ids":enc["input_ids"],
        "token_type_ids":enc["token_type_ids"],
        "labels":batch["label"]
    }

train_ds = dataset["train"].map(preprocess, batched=True)
test_ds = dataset["test"].map(preprocess,  batched=True)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
      self.ds = ds
    def __len__(self):
      return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return (
            torch.tensor(item["input_ids"]),
            torch.tensor(item["token_type_ids"]),
            torch.tensor(item["labels"])
        )

train_dl = DataLoader(NewsDataset(train_ds), batch_size=32, shuffle=True)
test_dl = DataLoader(NewsDataset(test_ds),  batch_size=32)

# Fine-tuning model
num_labels = 4
class BertForSequenceClassification(nn.Module):
    def __init__(self, config, pretrained_bert):
        super().__init__()
        self.bert       = pretrained_bert
        self.classifier = nn.Linear(config.d_model, num_labels)

    def forward(self, input_ids, segment_ids):
        _, pooled = self.bert(input_ids, segment_ids)
        return self.classifier(pooled)

pretrained_bert = results["model"].bert.to(device)

# Fine-tuning setting
model = BertForSequenceClassification(config, pretrained_bert).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_dl) * 8  # 8 epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
               num_warmup_steps=500,
               num_training_steps=total_steps)

# Fine-tuning
train_losses, train_accs = [], []
for epoch in range(8):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for input_ids, token_type_ids, labels in tqdm(train_dl, desc=f"FT Epoch {epoch+1}"):
        input_ids, token_type_ids, labels = (
            input_ids.to(device),
            token_type_ids.to(device),
            labels.to(device)
        )
        optimizer.zero_grad()
        logits = model(input_ids, token_type_ids)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=-1)
        epoch_acc += (preds == labels).float().mean().item()

    train_losses.append(epoch_loss / len(train_dl))
    train_accs.append(epoch_acc / len(train_dl))
    print(f"[Epoch {epoch+1}] Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f}")

# Evaluation
model.eval()
test_acc = 0.0
with torch.no_grad():
    for input_ids, token_type_ids, labels in test_dl:
        input_ids, token_type_ids, labels = (
            input_ids.to(device),
            token_type_ids.to(device),
            labels.to(device)
        )
        logits = model(input_ids, token_type_ids)
        preds = logits.argmax(dim=-1)
        test_acc += (preds == labels).float().mean().item()

print(f"Test Accuracy: {test_acc/len(test_dl):.4f}")

# Visualization
plt.figure(figsize=(8,4))
plt.plot(range(1,9), train_losses, label="Train Loss")
plt.plot(range(1,9), train_accs,   label="Train Acc")
plt.xlabel("Epoch")
plt.legend()
plt.show()
