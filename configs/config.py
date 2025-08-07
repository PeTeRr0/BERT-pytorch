class BERTConfig:
    def __init__(self):
        self.vocab_size = 30522              # BERT-base vocab size
        self.d_model = 768                   # hidden size
        self.num_attention_heads = 12       # num heads
        self.num_hidden_layers = 12          # encoder layers
        self.max_position_embeddings = 512  # max position embeddings
        self.num_segments = 2                # token type segments (A,B)
        self.hidden_dropout_prob = 0.1       # dropout rate
        self.intermediate_size = 3072         # feed forward hidden dim
        self.learning_rate = 5e-5            # learning rate
        self.weight_decay = 0.01             # weight decay (the paper uses Adam)
        self.eps = 1e-6                      # adam epsilon
        self.batch_size = 32                 # batch size
        self.max_epochs = 47                  # epoch (the paper uses over 40 epoch)
        self.warmup_steps = 10000            # warmup steps (the paper has 10k steps)