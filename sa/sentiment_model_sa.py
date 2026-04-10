import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        """
        x: (batch_size, seq_len)
        lengths: (batch_size,)
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Como es bidireccional:
        # hidden[0] = última salida forward
        # hidden[1] = última salida backward
        hidden_forward = hidden[0]
        hidden_backward = hidden[1]

        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden)  # (batch_size, num_classes)

        return logits