import torch
import torch.nn as nn

class MAEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids):
        emb = self.embeddings(input_ids)
        logits = self.decoder(emb)
        return logits