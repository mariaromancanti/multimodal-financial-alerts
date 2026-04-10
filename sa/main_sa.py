import os
import torch

from build_sa_vocab import generate_sa_vocab
from sa_data_utils import get_sa_dataloaders
from sentiment_model_sa import SentimentBiLSTM
from train_sa import run_sa_training
from evaluate_sa import evaluate_sa_model, predict_sentiment
from utils import load_json, files_exist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = "reduced_data_with_sentiment"
VOCAB_DIR = "vocab_sa"
MODEL_DIR = "models"

TRAIN_FILE = os.path.join(DATA_DIR, "train_with_sentiment.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "validation_with_sentiment.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test_with_sentiment.jsonl")

TOKEN_VOCAB_PATH = os.path.join(VOCAB_DIR, "token_to_idx_sa.json")
SENTIMENT_VOCAB_PATH = os.path.join(VOCAB_DIR, "sentiment_to_idx.json")

MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_sa.pt")


def ensure_directories():
    os.makedirs(VOCAB_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def ensure_data():
    required_files = [TRAIN_FILE, VAL_FILE, TEST_FILE]
    if not files_exist(required_files):
        raise FileNotFoundError(
            "No se encontraron los archivos con sentimiento en "
            f"{DATA_DIR}. Asegúrate de haber generado:\n"
            "- train_with_sentiment.jsonl\n"
            "- validation_with_sentiment.jsonl\n"
            "- test_with_sentiment.jsonl"
        )


def ensure_vocab():
    if files_exist([TOKEN_VOCAB_PATH, SENTIMENT_VOCAB_PATH]):
        print("Vocabulario de SA ya existe -> skip build_vocab")
        return

    generate_sa_vocab(
        train_path=TRAIN_FILE,
        vocab_dir=VOCAB_DIR,
        min_freq=1
    )


def build_model(vocab_size, num_classes):
    return SentimentBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)


def train_main():
    ensure_directories()
    ensure_data()
    ensure_vocab()

    token_to_idx = load_json(TOKEN_VOCAB_PATH)
    sentiment_to_idx = load_json(SENTIMENT_VOCAB_PATH)

    train_loader, val_loader = get_sa_dataloaders(
        train_path=TRAIN_FILE,
        val_path=VAL_FILE,
        token_to_idx=token_to_idx,
        sentiment_to_idx=sentiment_to_idx,
        batch_size=16
    )

    model = build_model(
        vocab_size=len(token_to_idx),
        num_classes=len(sentiment_to_idx)
    )

    run_sa_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        sentiment_to_idx=sentiment_to_idx,
        model_path=MODEL_PATH,
        device=device,
        epochs=20,
        lr=0.001
    )


def evaluate_test():
    ensure_data()
    ensure_vocab()

    token_to_idx = load_json(TOKEN_VOCAB_PATH)
    sentiment_to_idx = load_json(SENTIMENT_VOCAB_PATH)

    model = build_model(
        vocab_size=len(token_to_idx),
        num_classes=len(sentiment_to_idx)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    metrics = evaluate_sa_model(
        model=model,
        test_path=TEST_FILE,
        token_to_idx=token_to_idx,
        sentiment_to_idx=sentiment_to_idx,
        batch_size=16,
        device=device
    )

    print("\n=== RESULTADOS TEST ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])

    return metrics


def predict(text):
    ensure_vocab()

    token_to_idx = load_json(TOKEN_VOCAB_PATH)
    sentiment_to_idx = load_json(SENTIMENT_VOCAB_PATH)

    idx_to_sentiment = {idx: label for label, idx in sentiment_to_idx.items()}

    model = build_model(
        vocab_size=len(token_to_idx),
        num_classes=len(sentiment_to_idx)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    label, probs = predict_sentiment(
        model=model,
        text=text,
        token_to_idx=token_to_idx,
        idx_to_sentiment=idx_to_sentiment,
        device=device
    )

    print(f"Texto: {text}")
    print(f"Sentimiento predicho: {label}")
    print(f"Probabilidades: {probs}")

    return label, probs


if __name__ == "__main__":
    # Entrenar
    # train_main()

    # Evaluar en test
    evaluate_test()

    # Predicción individual
    # predict("The company reported strong quarterly earnings and improved guidance.")