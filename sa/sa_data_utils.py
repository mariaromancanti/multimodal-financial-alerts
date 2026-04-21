import json
from torch.utils.data import Dataset, DataLoader
import torch


class SADataset(Dataset):
    def __init__(self, path, token_to_idx, sentiment_to_idx):
        self.samples = []
        self.token_to_idx = token_to_idx
        self.sentiment_to_idx = sentiment_to_idx

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)

                if (
                    "tokens" not in sample
                    or "sentiment" not in sample
                    or not isinstance(sample["tokens"], list)
                    or not isinstance(sample["sentiment"], str)
                ):
                    continue

                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tokens = sample["tokens"]
        sentiment = sample["sentiment"].lower()

        unk_idx = self.token_to_idx["<UNK>"]

        token_ids = [
            self.token_to_idx.get(str(token), unk_idx)
            for token in tokens
        ]

        label_id = self.sentiment_to_idx[sentiment]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


def collate_sa(batch):
    sentences, labels = zip(*batch)

    lengths = torch.tensor([len(sentence) for sentence in sentences], dtype=torch.long)
    max_len = max(lengths).item()

    pad_idx = 0

    padded_sentences = []
    for sentence in sentences:
        padded = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded[:len(sentence)] = sentence
        padded_sentences.append(padded)

    padded_sentences = torch.stack(padded_sentences)
    labels = torch.stack(labels)

    return padded_sentences, labels, lengths


def get_sa_dataloaders(train_path, val_path, token_to_idx, sentiment_to_idx, batch_size=16):
    train_dataset = SADataset(train_path, token_to_idx, sentiment_to_idx)
    val_dataset = SADataset(val_path, token_to_idx, sentiment_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sa
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sa
    )

    return train_loader, val_loader
