import json
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "reduced_data"
OUTPUT_DIR = SCRIPT_DIR / "reduced_data_with_sentiment"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILES = {
    "train": "train_reduced.jsonl",
    "validation": "validation_reduced.jsonl",
    "test": "test_reduced.jsonl",
}

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
BATCH_SIZE = 16


POS_THRESHOLD = 0.3
NEG_THRESHOLD = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def detokenize(tokens):
    """
    Reconstrucción simple desde tokens.
    Para vuestro caso basta con unir con espacios.
    """
    return " ".join(map(str, tokens)).strip()


def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_batch(texts, tokenizer, model):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1).cpu()

    id2label = model.config.id2label
    label2id = {label.lower(): idx for idx, label in id2label.items()}

    pos_id = label2id["positive"]
    neg_id = label2id["negative"]
    neu_id = label2id["neutral"]

    labels = []
    scores = []

    for prob_vector in probs:
        p_positive = prob_vector[pos_id].item()
        p_negative = prob_vector[neg_id].item()
        p_neutral = prob_vector[neu_id].item()

        # Prioridad: positive > negative > neutral
        if p_positive >= POS_THRESHOLD:
            label = "positive"
            score = p_positive
        elif p_negative >= NEG_THRESHOLD:
            label = "negative"
            score = p_negative
        else:
            label = "neutral"
            score = p_neutral

        labels.append(label)
        scores.append(score)

    return labels, scores


def enrich_with_sentiment(data, tokenizer, model):
    enriched = []
    counter = Counter()

    texts = [detokenize(sample["tokens"]) for sample in data]

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Etiquetando con FinBERT"):
        batch_samples = data[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]

        labels, scores = predict_batch(batch_texts, tokenizer, model)

        for sample, text, label, score in zip(batch_samples, batch_texts, labels, scores):
            new_sample = dict(sample)
            new_sample["text"] = text
            new_sample["sentiment"] = label
            new_sample["sentiment_score"] = round(float(score), 4)

            enriched.append(new_sample)
            counter[label] += 1

    return enriched, counter


def main():
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando modelo: {MODEL_NAME}")
    print(f"Umbral positive: {POS_THRESHOLD}")
    print(f"Umbral negative: {NEG_THRESHOLD}")

    tokenizer, model = load_finbert()

    for split_name, filename in INPUT_FILES.items():
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / f"{split_name}_with_sentiment.jsonl"

        if not input_path.exists():
            print(f"No existe: {input_path}")
            continue

        print(f"\nProcesando {split_name}...")
        data = load_jsonl(input_path)

        data = [
            sample for sample in data
            if "tokens" in sample and "ner_tags" in sample
            and isinstance(sample["tokens"], list)
            and isinstance(sample["ner_tags"], list)
        ]

        enriched_data, counter = enrich_with_sentiment(data, tokenizer, model)
        save_jsonl(enriched_data, output_path)

        print(f"Guardado en: {output_path}")
        print(f"Distribución sentiment: {dict(counter)}")


if __name__ == "__main__":
    main()