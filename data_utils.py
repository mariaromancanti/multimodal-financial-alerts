import json
import torch
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    def __init__(self, path, vocab, tag_to_idx):
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                tokens = item["tokens"]
                tags = item["ner_tags"]

                token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
                tag_ids = [tag_to_idx[tag] for tag in tags]

                self.samples.append((token_ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    pad_token_id = 0
    pad_tag_id = -1

    sentences, tags = zip(*batch)
    max_len = max(len(sentence) for sentence in sentences)

    padded_sentences = []
    padded_tags = []

    for sentence, tag_seq in zip(sentences, tags):
        padded_sentences.append(
            sentence + [pad_token_id] * (max_len - len(sentence))
        )
        padded_tags.append(
            tag_seq + [pad_tag_id] * (max_len - len(tag_seq))
        )

    return (
        torch.tensor(padded_sentences, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
    )


def get_dataloaders(train_path, val_path, vocab, tag_to_idx, batch_size=16):
    train_dataset = NERDataset(train_path, vocab, tag_to_idx)
    val_dataset = NERDataset(val_path, vocab, tag_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader