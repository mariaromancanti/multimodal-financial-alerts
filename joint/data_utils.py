import json

import torch
from torch.utils.data import DataLoader, Dataset


class JointNersaDataset(Dataset):
    def __init__(self, path, token_to_idx, tag_to_idx, sentiment_to_idx):
        self.samples = []
        self.token_to_idx = token_to_idx
        self.tag_to_idx = tag_to_idx
        self.sentiment_to_idx = sentiment_to_idx

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                if not self._is_valid(sample):
                    continue

                token_ids = [
                    token_to_idx.get(str(token), token_to_idx["<UNK>"])
                    for token in sample["tokens"]
                ]
                tag_ids = [tag_to_idx[tag] for tag in sample["ner_tags"]]
                sentiment_id = sentiment_to_idx[sample["sentiment"].lower()]

                self.samples.append((token_ids, tag_ids, sentiment_id))

    def _is_valid(self, sample):
        return (
            isinstance(sample.get("tokens"), list)
            and isinstance(sample.get("ner_tags"), list)
            and isinstance(sample.get("sentiment"), str)
            and len(sample["tokens"]) == len(sample["ner_tags"])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, tag_ids, sentiment_id = self.samples[idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            torch.tensor(sentiment_id, dtype=torch.long),
        )


def collate_joint(batch):
    sentences, ner_tags, sa_labels = zip(*batch)

    lengths = torch.tensor([len(sentence) for sentence in sentences], dtype=torch.long)
    max_len = max(lengths).item()
    pad_idx = 0
    pad_tag_id = -1

    padded_sentences = []
    padded_tags = []

    for sentence, tags in zip(sentences, ner_tags):
        padded_sentence = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded_sentence[: len(sentence)] = sentence
        padded_sentences.append(padded_sentence)

        padded_tag_seq = torch.full((max_len,), pad_tag_id, dtype=torch.long)
        padded_tag_seq[: len(tags)] = tags
        padded_tags.append(padded_tag_seq)

    return (
        torch.stack(padded_sentences),
        torch.stack(padded_tags),
        torch.stack(sa_labels),
        lengths,
    )


def get_joint_dataloaders(
    train_path,
    val_path,
    token_to_idx,
    tag_to_idx,
    sentiment_to_idx,
    batch_size=16,
):
    train_dataset = JointNersaDataset(
        train_path, token_to_idx, tag_to_idx, sentiment_to_idx
    )
    val_dataset = JointNersaDataset(
        val_path, token_to_idx, tag_to_idx, sentiment_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_joint,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_joint,
    )

    return train_loader, val_loader
