import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions of different frequencies.
    This helps the model understand token positions in the sequence without
    relying on recurrence or convolution.
    
    Paper reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a vector of shape (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create division term for different dimensions
        # This creates frequencies that decrease geometrically from 2π to 10000·2π
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices in the array (0,2,4,...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the array (1,3,5,...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not parameters but should be saved and restored)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor with shape [batch_size, seq_length, d_model]
        Returns:
            Tensor with shape [batch_size, seq_length, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class MAEmbeddingModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim=768,
                 max_seq_length=512,
                 num_token_types=2,
                 dropout=0.1,
                 layer_norm_eps=1e-12):
        super().__init__()
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)
        
        # Positional embeddings
        self.position_embeddings = PositionalEncoding(emb_dim, max_seq_length)
        
        # Token type embeddings (useful for tasks like NSP or sentence pairs)
        self.token_type_embeddings = nn.Embedding(num_token_types, emb_dim)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.decoder = nn.Linear(emb_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, 
                input_ids, 
                token_type_ids=None,
                attention_mask=None):
        
        seq_length = input_ids.size(1)
        
        # Get token embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Add positional encodings
        embeddings = self.position_embeddings(embeddings)
        
        # Add token type embeddings if provided
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
            
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = embeddings * attention_mask
            
        # Get logits
        logits = self.decoder(embeddings)
        
        return logits
    
    def get_embeddings(self, 
                      input_ids, 
                      token_type_ids=None,
                      attention_mask=None,
                      pool_output=False):
        """
        Get embeddings without decoder projection
        Args:
            pool_output: If True, return mean pooled embeddings across sequence length
        """
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.position_embeddings(embeddings)
        
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
            
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = embeddings * attention_mask
            
        if pool_output:
            # Mean pool across sequence length
            if attention_mask is not None:
                embeddings = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                embeddings = embeddings.mean(1)
                
        return embeddings

    def save_pretrained(self, path):
        """Save model state dict and config"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.token_embeddings.num_embeddings,
                'emb_dim': self.token_embeddings.embedding_dim,
                'num_token_types': self.token_type_embeddings.num_embeddings
            }
        }, path)
    
    @classmethod
    def from_pretrained(cls, path):
        """Load model from saved state dict and config"""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model