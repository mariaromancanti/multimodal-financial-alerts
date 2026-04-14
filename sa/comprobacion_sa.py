import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path("reduced_data_with_sentiment")
FILES = [
    "train_with_sentiment.jsonl",
    "validation_with_sentiment.jsonl",
    "test_with_sentiment.jsonl",
]

VALID_SENTIMENTS = {"positive", "neutral", "negative"}


def analyze_file(path: Path):
    print("=" * 70)
    print(f"Comprobando: {path.name}")

    total = 0
    counter = Counter()
    bad_rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            row = json.loads(line)
            total += 1

            if "sentiment" not in row or row["sentiment"] not in VALID_SENTIMENTS:
                bad_rows.append(i)
                continue

            counter[row["sentiment"]] += 1

    print(f"Filas totales: {total}")
    print(f"Distribución: {dict(counter)}")

    if bad_rows:
        print(f"Filas problemáticas (primeras 10): {bad_rows[:10]}")
    else:
        print("Todo correcto.")
    print()


def main():
    for filename in FILES:
        path = BASE_DIR / filename
        if path.exists():
            analyze_file(path)
        else:
            print(f"No existe: {path}")


if __name__ == "__main__":
    main()