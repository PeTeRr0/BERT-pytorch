#@title Training and Evaluation

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

config = BERTConfig()

config.vocab_size = tokenizer.vocab_size

def train_and_eval(config, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreTrainedModel(config).to(device)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_nsp = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=config.eps, weight_decay=config.weight_decay)
    dataset = WikiTextDataset(tokenizer, seq_len=128, mlm_prob=0.15)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    total_steps = len(dataloader) * config.max_epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)

    mlm_losses, nsp_losses, tot_losses = [], [], []
    mlm_accs, nsp_accs = [], []

    for epoch in range(config.max_epochs):
        model.train()
        epoch_mlm_loss = epoch_nsp_loss = 0.0
        epoch_mlm_acc = epoch_nsp_acc = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids, segment_ids, mlm_labels, is_next = [x.to(device) for x in batch]
            optimizer.zero_grad()
            mlm_logits, nsp_logits = model(input_ids, segment_ids)
            mlm_loss = criterion_mlm(mlm_logits.view(-1, config.vocab_size), mlm_labels.view(-1))
            nsp_loss = criterion_nsp(nsp_logits, is_next)
            loss = mlm_loss + nsp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()

            epoch_mlm_loss += mlm_loss.item()
            epoch_nsp_loss += nsp_loss.item()
            # Accuracy
            mask = mlm_labels != -100
            mlm_acc = (mlm_logits.argmax(dim=-1)[mask] == mlm_labels[mask]).float().mean().item()
            nsp_acc = (nsp_logits.argmax(dim=-1) == is_next).float().mean().item()
            epoch_mlm_acc += mlm_acc
            epoch_nsp_acc += nsp_acc

        mlm_losses.append(epoch_mlm_loss / len(dataloader))
        nsp_losses.append(epoch_nsp_loss / len(dataloader))
        tot_losses.append((epoch_mlm_loss + epoch_nsp_loss) / len(dataloader))
        mlm_accs.append(epoch_mlm_acc / len(dataloader))
        nsp_accs.append(epoch_nsp_acc / len(dataloader))

        print(f"[Epoch {epoch+1}] MLM loss {mlm_losses[-1]:.4f}, NSP loss {nsp_losses[-1]:.4f}, MLM acc {mlm_accs[-1]:.4f}, NSP acc {nsp_accs[-1]:.4f}")

    # Visualization
    epochs = range(1, config.max_epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, mlm_losses, label='MLM Loss')
    plt.plot(epochs, nsp_losses, label='NSP Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, mlm_accs, label='MLM Accuracy')
    plt.plot(epochs, nsp_accs, label='NSP Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "mlm_losses": mlm_losses,
        "nsp_losses": nsp_losses,
        "mlm_accs": mlm_accs,
        "nsp_accs": nsp_accs,
        "tot_losses": tot_losses,
        "model": model
    }

results = train_and_eval(config, tokenizer)

# Calculate Perplexity from MLM Loss
final_mlm_loss = results["mlm_losses"][-1]
perplexity = math.exp(final_mlm_loss)

print(f"MLM Loss: {results['mlm_losses'][-1]:.4f}")
print(f"NSP Loss: {results['nsp_losses'][-1]:.4f}")
print(f"MLM Accuracy: {results['mlm_accs'][-1]:.4f}")
print(f"NSP Accuracy: {results['nsp_accs'][-1]:.4f}")
print(f"Perplexity (MLM): {perplexity:.4f}")