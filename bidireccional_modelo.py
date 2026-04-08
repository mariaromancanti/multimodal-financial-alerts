import torch
import torch.nn as nn


class BiLSTMNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        tagset_size: int,
        padding_idx: int = 0,
        dropout: float = 0.2
    ) -> None:
        
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=tagset_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                  
        lstm_out, _ = self.lstm(emb)            
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)    
        return logits