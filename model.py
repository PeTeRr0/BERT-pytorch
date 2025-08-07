#@title BERT

import torch
from torch import nn
from typing import Tuple
import math

class Attention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    self.dropout = nn.Dropout(dropout)

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)
  def forward(self, x):
    b, s, _ = x.size() # b = batch_size, # s = sequence length
    # (b, s, num_heads, d_k) → (b, num_heads, s, d_k)
    q = self.w_q(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)

    # scaling
    attn_out = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.d_k)
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, s, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    self.dropout = nn.Dropout(dropout)

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

  def forward(self, q, k, v):
    b, q_len, _ = q.size() # b = batch_size, s = seq_len
    b, kv_len, _ = k.size()
    # (bs, seq, num_heads, d_k) → (bs, num_heads, seq, d_k)
    q = self.w_q(q).view(b, q_len, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(k).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(v).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)

    # scaling dividing by math.sqrt(self.d_k)
    attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, q_len, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

class Encoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.attn = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.layer_norm1 = nn.LayerNorm(d_model)

    self.ffn = FeedForward(d_model, num_heads, d_ff, dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(d_model)

  def forward(self, x):
    attn_out = self.attn(x, x, x)
    x = x + self.dropout1(attn_out)
    x = self.layer_norm1(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout2(ffn_out)
    x = self.layer_norm2(x)

    return x

class EncoderStack(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.layers = nn.ModuleList([
        Encoder(d_model, num_heads, d_ff, dropout)
        for _ in range(num_layers)
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class FeedForward(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)

    return x

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    # Token embedding table (vocab_size * d_model)
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    # Optional
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

  def forward(self, token_ids):
    return self.token_embedding(token_ids)

class SegmentEmbedding(nn.Module):
  def __init__(self, num_segments, d_model):
    super().__init__()
    # Segment embedding table (num_segments * d_model)
    self.segment_embedding = nn.Embedding(num_segments, d_model)
    # Optional
    nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)

  def forward(self, segment_ids):
    return self.segment_embedding(segment_ids)

class PositionEmbedding(nn.Module):
  def __init__(self, max_position_embedding, d_model):
    super().__init__()
    # Position embedding table (max_position_embedding* d_model)
    self.position_embedding = nn.Embedding(max_position_embedding, d_model)
    # Optional
    nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

  def forward(self, input_ids):
    """
    input_ids: (batch_size, seq_len) posiiton information extraction
    return: (batch_size, seq_len, d_model) position embedding
    """
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    # Generate position index from 0 to seq_len-1
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    # (seq_len, ) -> (batch_size, seq_len)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    return self.position_embedding(position_ids)

class Embeddings(nn.Module):
  def __init__(self, vocab_size, d_model, max_position_embedding, num_segments, dropout):
    super().__init__()
    self.token_embedding = TokenEmbedding(vocab_size, d_model)
    self.segment_embedding = SegmentEmbedding(num_segments, d_model)
    self.position_embedding = PositionEmbedding(max_position_embedding, d_model)
    self.layer_norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, token_ids, segment_ids, input_ids):
    x = self.token_embedding(token_ids) + self.segment_embedding(segment_ids) + self.position_embedding(input_ids)
    x = self.layer_norm(x)
    x = self.dropout(x)

    return x

class Pooler(nn.Module): # We take the final hidden state of the first token ([CLS]) as the aggregate representation for classification tasks
  def __init__(self, d_model):
    super().__init__()
    self.dense = nn.Linear(d_model, d_model)
    self.activation = nn.Tanh()

  def forward(self, hidden_states):
    # hidden_states: (batch, seq_len, d_model)
    cls_token = hidden_states[:, 0] # (batch, d_model)
    cls_token = self.dense(cls_token)
    cls_token = self.activation(cls_token)

    return cls_token

class Model(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = Embeddings(
            config.vocab_size, config.d_model,
            config.max_position_embeddings,
            config.num_segments,
            config.hidden_dropout_prob
        )
    self.encoder = EncoderStack(
            config.num_hidden_layers,
            config.d_model,
            config.num_attention_heads,
            config.intermediate_size, # d_ff
            config.hidden_dropout_prob
        )
    self.pooler = Pooler(config.d_model)

  def forward(self, token_ids, segment_ids):
    x = self.embeddings(token_ids, segment_ids, token_ids)
    x = self.encoder(x)
    pooled = self.pooler(x)

    return x, pooled

class MLM(nn.Module): # Masked LM
  def __init__(self, config, token_embedding_weights):
    super().__init__()
    self.dense = nn.Linear(config.d_model, config.d_model)
    self.activation = nn.GELU()
    self.layer_norm = nn.LayerNorm(config.d_model)
    # decoder: tied weight
    self.predictions = nn.Linear(config.d_model, config.vocab_size, bias=True)
    self.predictions.weight = token_embedding_weights
    self.bias = nn.Parameter(torch.zeros(config.vocab_size))

  def forward(self, hidden_states):
    x = self.dense(hidden_states)
    x = self.activation(x)
    x = self.layer_norm(x)
    x = self.predictions(x) + self.bias

    return x  # (batch, seq_len, vocab_size)

class NSP(nn.Module): # Next Sentence Prediction
  def __init__(self, config):
      super().__init__()
      self.classifier = nn.Linear(config.d_model, 2)

  def forward(self, pooled_output):
      return self.classifier(pooled_output)  # (batch, 2)

class PreTrainedModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.bert = Model(config)
    self.mlm = MLM(config, self.bert.embeddings.token_embedding.token_embedding.weight)
    self.nsp = NSP(config)

  def forward(self, token_ids, segment_ids):
    sequence_output, pooled_output = self.bert(token_ids, segment_ids)
    mlm_logits = self.mlm(sequence_output)
    nsp_logits = self.nsp(pooled_output)

    return mlm_logits, nsp_logits

def mask_tokens(
    input_ids: torch.LongTensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float=0.15,
    special_tokens_mask: torch.BoolTensor=None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
  """
  input_ids: torch.LongTensor of shape (batch_size, seq_len)
  mask_token_id: id of [MASK] token
  vocab_size: size of vocabulary
  mlm_probability: probability to mask each token
  special_tokens_mask: optional BoolTensor same shape as input_ids,
                        True where token is special ([CLS],[SEP],pad,…)
  returns:
      masked_input_ids: with masked/replaced tokens
      labels: with original token ids at masked positions, -100 elsewhere
  """
  # Make a copy of the input IDs to use as labels for MLM
  labels = input_ids.clone()

  # Select positions to mask
  # Create a matrix filled with the masking probability (e.g., 0.15) for every token
  probability_matrix = torch.full(labels.shape, mlm_probability)
  # If we have a mask for special tokens (e.g., [CLS], [SEP], [PAD]), set their probability to 0
  if special_tokens_mask is not None:
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  # Sample a mask decision for each token (True = will be masked)
  masked_indices = torch.bernoulli(probability_matrix).bool()

  # For tokens that are not selected for masking, set their label to -100 so loss ignores them
  labels[~masked_indices] = -100

  # Replace the selected (masked) tokens

  # 1) 80%, replace masked tokens with the [MASK] token
  # Sample decisions for replacement (True = replace with [MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  input_ids[indices_replaced] = mask_token_id

  # 2) 10%, replace masked tokens with a random token
  # Sample decisions for random replacement (half of the remaining masked positions)
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  # Draw random token IDs from the full vocabulary
  random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
  input_ids[indices_random] = random_tokens[indices_random]

  # 3) 10%, unchanged tokens

  # Return the modified input IDs (with masks/random tokens) and the MLM labels
  return input_ids, labels
