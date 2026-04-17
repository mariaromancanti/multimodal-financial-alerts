import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from joint.data_utils import get_joint_dataloaders
from joint.multitask_model import JointBiLSTMNERSA
from ner.utils import load_vocabularies
from sa.utils import load_json


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER_TOKEN_VOCAB_PATH = os.path.join(PROJECT_ROOT, "vocab", "token_to_idx.json")
NER_TAG_VOCAB_PATH = os.path.join(PROJECT_ROOT, "vocab", "tag_to_idx.json")
SA_SENTIMENT_VOCAB_PATH = os.path.join(PROJECT_ROOT, "sa", "vocab_sa", "sentiment_to_idx.json")

JOINT_TRAIN_PATH = os.path.join(PROJECT_ROOT, "sa", "train_with_sentiment.jsonl")
JOINT_VAL_PATH = os.path.join(PROJECT_ROOT, "sa", "validation_with_sentiment.jsonl")
JOINT_MODEL_PATH = os.path.join(PROJECT_ROOT, "joint", "models", "bilstm_joint_ner_sa.pt")


def compute_sa_class_weights(train_loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float)

    for _, _, sa_labels, _ in train_loader:
        for label in sa_labels:
            counts[label.item()] += 1

    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def build_joint_model(vocab_size, ner_tagset_size, sentiment_num_classes, device):
    return JointBiLSTMNERSA(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        ner_tagset_size=ner_tagset_size,
        sentiment_num_classes=sentiment_num_classes,
        padding_idx=0,
        dropout=0.2,
    ).to(device)


def _run_epoch(model, data_loader, optimizer, ner_criterion, sa_criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ner_loss = 0.0
    total_sa_loss = 0.0
    ner_correct = 0
    ner_total = 0
    all_sa_preds = []
    all_sa_labels = []

    grad_context = torch.enable_grad if train else torch.no_grad
    with grad_context():
        for sentences, ner_tags, sa_labels, lengths in data_loader:
            sentences = sentences.to(device)
            ner_tags = ner_tags.to(device)
            sa_labels = sa_labels.to(device)
            lengths = lengths.to(device)

            if train:
                optimizer.zero_grad()

            ner_logits, sa_logits = model(sentences, lengths)
            ner_loss = ner_criterion(
                ner_logits.reshape(-1, ner_logits.size(-1)),
                ner_tags.reshape(-1),
            )
            sa_loss = sa_criterion(sa_logits, sa_labels)
            loss = ner_loss + sa_loss

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_ner_loss += ner_loss.item()
            total_sa_loss += sa_loss.item()

            ner_preds = torch.argmax(ner_logits, dim=-1)
            valid_mask = ner_tags != -1
            ner_correct += (ner_preds[valid_mask] == ner_tags[valid_mask]).sum().item()
            ner_total += valid_mask.sum().item()

            sa_preds = torch.argmax(sa_logits, dim=1)
            all_sa_preds.extend(sa_preds.detach().cpu().tolist())
            all_sa_labels.extend(sa_labels.detach().cpu().tolist())

    avg_total_loss = total_loss / len(data_loader)
    avg_ner_loss = total_ner_loss / len(data_loader)
    avg_sa_loss = total_sa_loss / len(data_loader)
    ner_accuracy = ner_correct / ner_total if ner_total > 0 else 0.0
    sa_accuracy = accuracy_score(all_sa_labels, all_sa_preds)
    sa_f1 = f1_score(all_sa_labels, all_sa_preds, average="macro")

    return {
        "loss": avg_total_loss,
        "ner_loss": avg_ner_loss,
        "sa_loss": avg_sa_loss,
        "ner_accuracy": ner_accuracy,
        "sa_accuracy": sa_accuracy,
        "sa_f1": sa_f1,
    }


def run_joint_training(
    model,
    train_loader,
    val_loader,
    model_path,
    device,
    sentiment_to_idx,
    epochs=10,
    lr=0.001,
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    ner_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    sa_class_weights = compute_sa_class_weights(
        train_loader, len(sentiment_to_idx), device
    )
    sa_criterion = nn.CrossEntropyLoss(weight=sa_class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    print("Joint SA class weights:", sa_class_weights.detach().cpu().tolist())

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            ner_criterion=ner_criterion,
            sa_criterion=sa_criterion,
            device=device,
            train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            data_loader=val_loader,
            optimizer=optimizer,
            ner_criterion=ner_criterion,
            sa_criterion=sa_criterion,
            device=device,
            train=False,
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(
            "Train -> "
            f"loss: {train_metrics['loss']:.4f} | "
            f"ner_loss: {train_metrics['ner_loss']:.4f} | "
            f"sa_loss: {train_metrics['sa_loss']:.4f} | "
            f"ner_acc: {train_metrics['ner_accuracy']:.4f} | "
            f"sa_acc: {train_metrics['sa_accuracy']:.4f} | "
            f"sa_f1: {train_metrics['sa_f1']:.4f}"
        )
        print(
            "Val   -> "
            f"loss: {val_metrics['loss']:.4f} | "
            f"ner_loss: {val_metrics['ner_loss']:.4f} | "
            f"sa_loss: {val_metrics['sa_loss']:.4f} | "
            f"ner_acc: {val_metrics['ner_accuracy']:.4f} | "
            f"sa_acc: {val_metrics['sa_accuracy']:.4f} | "
            f"sa_f1: {val_metrics['sa_f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), model_path)
            print(f"Best joint model saved to: {model_path}")


def ensure_joint_model(device, force_retrain=False):
    if os.path.exists(JOINT_MODEL_PATH) and not force_retrain:
        print(f"Using existing joint NER+SA model: {JOINT_MODEL_PATH}")
        return JOINT_MODEL_PATH

    print("Training joint NER+SA model with combined loss...")
    token_to_idx, tag_to_idx, _ = load_vocabularies(
        NER_TOKEN_VOCAB_PATH, NER_TAG_VOCAB_PATH
    )
    sentiment_to_idx = load_json(SA_SENTIMENT_VOCAB_PATH)

    train_loader, val_loader = get_joint_dataloaders(
        train_path=JOINT_TRAIN_PATH,
        val_path=JOINT_VAL_PATH,
        token_to_idx=token_to_idx,
        tag_to_idx=tag_to_idx,
        sentiment_to_idx=sentiment_to_idx,
        batch_size=16,
    )

    model = build_joint_model(
        vocab_size=len(token_to_idx),
        ner_tagset_size=len(tag_to_idx),
        sentiment_num_classes=len(sentiment_to_idx),
        device=device,
    )

    run_joint_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=JOINT_MODEL_PATH,
        device=device,
        sentiment_to_idx=sentiment_to_idx,
        epochs=10,
        lr=0.001,
    )

    return JOINT_MODEL_PATH


@torch.no_grad()
def predict_joint(model, text, token_to_idx, idx_to_tag, idx_to_sentiment, device):
    model.eval()

    tokens = text.split()
    token_ids = [token_to_idx.get(token, token_to_idx["<UNK>"]) for token in tokens]
    if len(token_ids) == 0:
        raise ValueError("El texto está vacío y no se puede predecir.")

    sentence_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(device)

    ner_logits, sa_logits = model(sentence_tensor, lengths)

    ner_pred_ids = ner_logits.argmax(dim=-1)[0]
    ner_output = [
        (token, idx_to_tag[tag_id.item()])
        for token, tag_id in zip(tokens, ner_pred_ids)
    ]

    sa_probs = torch.softmax(sa_logits, dim=1).squeeze(0)
    sa_pred_idx = torch.argmax(sa_probs).item()
    sa_label = idx_to_sentiment[sa_pred_idx]
    sa_output = [
        sa_label,
        {
            idx_to_sentiment[i]: round(float(sa_probs[i].item()), 4)
            for i in range(len(idx_to_sentiment))
        },
    ]

    return ner_output, sa_output
