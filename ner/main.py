import os
import torch

from PROCESING_DATOS import generate_reduced_data
from build_ner_vocab import generate_vocab
from bidireccional_modelo import BiLSTMNER
from evaluate_nuestro import predict_ner
from data_utils import get_dataloaders, NERDataset, collate_fn
from utils import load_vocabularies, files_exist
from train import run_training
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
VOCAB_DIR = "vocab"
MODEL_DIR = "models"

TRAIN_FILE = os.path.join(DATA_DIR, "train_reduced.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "validation_reduced.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test_reduced.jsonl")

TOKEN_VOCAB_PATH = os.path.join(VOCAB_DIR, "token_to_idx.json")
TAG_VOCAB_PATH = os.path.join(VOCAB_DIR, "tag_to_idx.json")

MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_ner.pt")


def ensure_data():
    if files_exist([TRAIN_FILE, VAL_FILE, TEST_FILE]):
        print("Datos ya existen -> skip processing")
        return

    generate_reduced_data(
        input_dir=".",
        output_dir=DATA_DIR,
        force=False
    )


def ensure_vocab():
    if files_exist([TOKEN_VOCAB_PATH, TAG_VOCAB_PATH]):
        print("Vocab ya existe -> skip build_vocab")
        return

    generate_vocab(
        data_dir=DATA_DIR,
        vocab_dir=VOCAB_DIR,
        force=False
    )


def build_model(vocab_size, tagset_size):
    return BiLSTMNER(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        tagset_size=tagset_size,
        dropout=0.2
    ).to(device)


def predict(text):
    vocab, tag_to_idx, idx_to_tag = load_vocabularies(
        TOKEN_VOCAB_PATH,
        TAG_VOCAB_PATH
    )

    model = build_model(len(vocab), len(tag_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    if "<OOV>" not in vocab and "<UNK>" in vocab:
        vocab["<OOV>"] = vocab["<UNK>"]

    result = predict_ner(model, text, vocab, idx_to_tag, device)

    for word, tag in result:
        print(f"{word}: {tag}")

    return result


def evaluate_test_simple(batch_size=16):
    vocab, tag_to_idx, _ = load_vocabularies(
        TOKEN_VOCAB_PATH,
        TAG_VOCAB_PATH
    )

    test_dataset = NERDataset(TEST_FILE, vocab, tag_to_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_model(len(vocab), len(tag_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for sentences, tags in test_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            logits = model(sentences)
            preds = torch.argmax(logits, dim=-1)

            mask = tags != -1
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def train_main():
    ensure_data()
    ensure_vocab()

    vocab, tag_to_idx, _ = load_vocabularies(
        TOKEN_VOCAB_PATH,
        TAG_VOCAB_PATH
    )

    train_loader, val_loader = get_dataloaders(
        train_path=TRAIN_FILE,
        val_path=VAL_FILE,
        vocab=vocab,
        tag_to_idx=tag_to_idx,
        batch_size=16
    )

    model = build_model(
        vocab_size=len(vocab),
        tagset_size=len(tag_to_idx)
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=MODEL_PATH,
        device=device,
        epochs=20,
        lr=0.001
    )


if __name__ == "__main__":
    evaluate_test_simple()

    # Para entrenar:
    # train_main()

    # Para predecir una frase:
    # predict("Revenues increased and InterestExpense decreased .")