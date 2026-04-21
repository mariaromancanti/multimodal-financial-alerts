import json
import os
from collections import Counter


SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_token_vocab(data, min_freq=1):
    counter = Counter()

    for sample in data:
        tokens = sample.get("tokens", [])
        for token in tokens:
            counter[str(token)] += 1

    token_to_idx = dict(SPECIAL_TOKENS)
    next_idx = len(token_to_idx)

    for token, freq in counter.items():
        if freq >= min_freq and token not in token_to_idx:
            token_to_idx[token] = next_idx
            next_idx += 1

    return token_to_idx


def build_sentiment_vocab():
    return {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }


def generate_sa_vocab(train_path, vocab_dir, min_freq=1):
    os.makedirs(vocab_dir, exist_ok=True)

    print(f"Cargando datos de entrenamiento desde: {train_path}")
    data = load_jsonl(train_path)

    data = [
        sample for sample in data
        if "tokens" in sample and "sentiment" in sample
        and isinstance(sample["tokens"], list)
        and isinstance(sample["sentiment"], str)
    ]

    print(f"Número de muestras válidas: {len(data)}")

    token_to_idx = build_token_vocab(data, min_freq=min_freq)
    sentiment_to_idx = build_sentiment_vocab()

    token_vocab_path = os.path.join(vocab_dir, "token_to_idx_sa.json")
    sentiment_vocab_path = os.path.join(vocab_dir, "sentiment_to_idx.json")

    save_json(token_to_idx, token_vocab_path)
    save_json(sentiment_to_idx, sentiment_vocab_path)

    print(f"Vocabulario de tokens guardado en: {token_vocab_path}")
    print(f"Tamaño vocabulario tokens: {len(token_to_idx)}")

    print(f"Vocabulario de sentimiento guardado en: {sentiment_vocab_path}")
    print(f"Clases de sentimiento: {sentiment_to_idx}")


if __name__ == "__main__":
    generate_sa_vocab(
        train_path="reduced_data_with_sentiment/train_with_sentiment.jsonl",
        vocab_dir="vocab_sa",
        min_freq=1
    )
