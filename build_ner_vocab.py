import json
from pathlib import Path
from collections import Counter

# CONFIG
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "reduced_data" / "train_reduced.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "vocabularies"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_FREQ = 1   # si luego queremos filtrar palabras raras, hay que subir esto a 2 o 3
LOWERCASE = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# LOAD DATA
def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# BUILD VOCABS
def normalize_token(token: str) -> str:
    return token.lower() if LOWERCASE else token

def build_word_vocab(data):
    counter = Counter()

    for sample in data:
        tokens = sample["tokens"]
        for token in tokens:
            counter[normalize_token(token)] += 1

    # tokens especiales al principio
    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    # añadimos el resto por frecuencia descendente
    sorted_words = sorted(
        [word for word, freq in counter.items() if freq >= MIN_FREQ],
        key=lambda w: (-counter[w], w)
    )

    for word in sorted_words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, counter

def build_label_vocab(data):
    labels = set()

    for sample in data:
        for tag in sample["ner_tags"]:
            labels.add(tag)

    # importante: O primero, luego el resto ordenado
    sorted_labels = ["O"] + sorted(label for label in labels if label != "O")

    label2idx = {label: idx for idx, label in enumerate(sorted_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    return label2idx, idx2label


# SAVE
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# MAIN
def main():
    data = load_jsonl(DATA_PATH)

    word2idx, idx2word, counter = build_word_vocab(data)
    label2idx, idx2label = build_label_vocab(data)

    save_json(word2idx, OUTPUT_DIR / "word2idx.json")
    save_json(idx2word, OUTPUT_DIR / "idx2word.json")
    save_json(label2idx, OUTPUT_DIR / "label2idx.json")
    save_json(idx2label, OUTPUT_DIR / "idx2label.json")

    print("Vocabularios NER generados correctamente")
    print(f"Guardados en: {OUTPUT_DIR}")
    print(f"Tamaño vocabulario de palabras: {len(word2idx)}")
    print(f"Número de etiquetas NER: {len(label2idx)}")
    print("\nTop 20 palabras más frecuentes:")
    for word, freq in counter.most_common(20):
        print(f"  {word}: {freq}")

    print("\nEtiquetas NER:")
    for label, idx in label2idx.items():
        print(f"  {label}: {idx}")

if __name__ == "__main__":
    main()