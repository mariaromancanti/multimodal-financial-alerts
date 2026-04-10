import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sa_data_utils import SADataset, collate_sa


@torch.no_grad()
def evaluate_sa_model(model, test_path, token_to_idx, sentiment_to_idx, batch_size, device):
    test_dataset = SADataset(test_path, token_to_idx, sentiment_to_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sa
    )

    model.eval()

    all_preds = []
    all_labels = []

    for sentences, labels, lengths in test_loader:
        sentences = sentences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(sentences, lengths)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    idx_to_sentiment = {idx: label for label, idx in sentiment_to_idx.items()}
    ordered_labels = [idx_to_sentiment[i] for i in range(len(idx_to_sentiment))]

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=ordered_labels,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm
    }


def simple_tokenize(text):
    """
    Tokenización simple para predicción manual.
    Si luego queréis algo más fino, esto se puede mejorar.
    """
    return text.strip().split()


@torch.no_grad()
def predict_sentiment(model, text, token_to_idx, idx_to_sentiment, device):
    model.eval()

    tokens = simple_tokenize(text)
    unk_idx = token_to_idx["<UNK>"]

    token_ids = [
        token_to_idx.get(token, unk_idx)
        for token in tokens
    ]

    if len(token_ids) == 0:
        raise ValueError("El texto está vacío y no se puede predecir sentimiento.")

    sentence_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(device)

    logits = model(sentence_tensor, lengths)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = torch.argmax(probs).item()
    pred_label = idx_to_sentiment[pred_idx]

    prob_dict = {
        idx_to_sentiment[i]: round(float(probs[i].item()), 4)
        for i in range(len(idx_to_sentiment))
    }

    return pred_label, prob_dict