#@title Pretraining Dataset: WikiText-2

from datasets import load_dataset
from torch.utils.data import Dataset

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, mlm_prob=0.15):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []

        for doc in dataset["text"]:
            if len(doc) < 50:
                continue
            tok = tokenizer(doc, truncation=True, max_length=seq_len-2, return_attention_mask=False)["input_ids"]
            tok = [tokenizer.cls_token_id] + tok + [tokenizer.sep_token_id]
            if len(tok) < seq_len:
                tok += [tokenizer.pad_token_id] * (seq_len - len(tok))
            self.examples.append(tok)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        segment_ids = torch.zeros_like(input_ids, dtype=torch.long)
        special_mask = (input_ids == self.tokenizer.cls_token_id) | (input_ids == self.tokenizer.sep_token_id) | (input_ids == self.tokenizer.pad_token_id)
        masked_input_ids, mlm_labels = mask_tokens(input_ids.clone(), mask_token_id=self.tokenizer.mask_token_id, vocab_size=self.tokenizer.vocab_size, special_tokens_mask=special_mask)
        is_next_label = torch.randint(0, 2, (1,), dtype=torch.long).item()
        return masked_input_ids, segment_ids, mlm_labels, is_next_label
