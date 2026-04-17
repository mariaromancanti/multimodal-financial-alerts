import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class JointBiLSTMNERSA(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        ner_tagset_size,
        sentiment_num_classes,
        padding_idx=0,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.ner_classifier = nn.Linear(hidden_dim * 2, ner_tagset_size)
        self.sa_classifier = nn.Linear(hidden_dim * 2, sentiment_num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

        lstm_out = self.dropout(lstm_out)
        ner_logits = self.ner_classifier(lstm_out)

        hidden_forward = hidden[0]
        hidden_backward = hidden[1]
        sentence_repr = torch.cat((hidden_forward, hidden_backward), dim=1)
        sentence_repr = self.dropout(sentence_repr)
        sa_logits = self.sa_classifier(sentence_repr)

        return ner_logits, sa_logits
