import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for sentences, labels, lengths in train_loader:
        sentences = sentences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        logits = model(sentences, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1_macro


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for sentences, labels, lengths in data_loader:
        sentences = sentences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(sentences, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1_macro


def compute_class_weights(train_loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float)

    for _, labels, _ in train_loader:
        for label in labels:
            counts[label.item()] += 1

    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def run_sa_training(
    model,
    train_loader,
    val_loader,
    sentiment_to_idx,
    model_path,
    device,
    epochs=20,
    lr=1e-3
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    num_classes = len(sentiment_to_idx)
    class_weights = compute_class_weights(train_loader, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0

    print("Pesos de clase:", class_weights.detach().cpu().tolist())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(
            f"Train -> loss: {train_loss:.4f} | "
            f"acc: {train_acc:.4f} | "
            f"f1_macro: {train_f1:.4f}"
        )
        print(
            f"Val   -> loss: {val_loss:.4f} | "
            f"acc: {val_acc:.4f} | "
            f"f1_macro: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"Mejor modelo guardado en: {model_path}")

    print(f"\nEntrenamiento finalizado. Mejor F1 macro en validación: {best_val_f1:.4f}")