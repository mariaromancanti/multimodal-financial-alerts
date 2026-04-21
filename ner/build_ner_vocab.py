import os
import json
from collections import Counter


def generate_vocab(data_dir="data", vocab_dir="vocab", force=False):
    os.makedirs(vocab_dir, exist_ok=True)

    token_path = os.path.join(vocab_dir, "token_to_idx.json")
    char_path = os.path.join(vocab_dir, "char_to_idx.json")
    tag_path = os.path.join(vocab_dir, "tag_to_idx.json")

    if not force and all(os.path.exists(p) for p in [token_path, char_path, tag_path]):
        print("Vocabularios ya existen. Se omite build_vocab.")
        return

    print("Construyendo vocabularios...")

    token_counter = Counter()
    char_counter = Counter()
    tag_set = set()

    files = [
        "train_reduced.jsonl",
        "validation_reduced.jsonl",
        "test_reduced.jsonl"
    ]

    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)

                tokens = sample["tokens"]
                tags = sample["ner_tags"]

                token_counter.update(tokens)
                tag_set.update(tags)

                for token in tokens:
                    char_counter.update(token)

    token_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in token_counter.items():
        token_to_idx[token] = len(token_to_idx)

    char_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for char, _ in char_counter.items():
        char_to_idx[char] = len(char_to_idx)

    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}

    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_to_idx, f)

    with open(char_path, "w", encoding="utf-8") as f:
        json.dump(char_to_idx, f)

    with open(tag_path, "w", encoding="utf-8") as f:
        json.dump(tag_to_idx, f)

    print("Vocabularios guardados en", vocab_dir)
