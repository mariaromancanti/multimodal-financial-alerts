import os
import torch


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device
):
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct_tokens = 0
        train_total_tokens = 0

        for sentences, tags in train_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()

            logits = model(sentences)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tags.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            valid_mask = tags != -1

            train_correct_tokens += (
                (predictions[valid_mask] == tags[valid_mask]).sum().item()
            )
            train_total_tokens += valid_mask.sum().item()

        avg_train_loss = train_loss_sum / len(train_loader)
        train_accuracy = (
            train_correct_tokens / train_total_tokens
            if train_total_tokens > 0 else 0.0
        )

        model.eval()
        val_loss_sum = 0.0
        val_correct_tokens = 0
        val_total_tokens = 0

        with torch.no_grad():
            for sentences, tags in val_loader:
                sentences = sentences.to(device)
                tags = tags.to(device)

                logits = model(sentences)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tags.reshape(-1)
                )

                val_loss_sum += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                valid_mask = tags != -1

                val_correct_tokens += (
                    (predictions[valid_mask] == tags[valid_mask]).sum().item()
                )
                val_total_tokens += valid_mask.sum().item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_accuracy = (
            val_correct_tokens / val_total_tokens
            if val_total_tokens > 0 else 0.0
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


def run_training(
    model,
    train_loader,
    val_loader,
    model_path,
    device,
    epochs=20,
    lr=0.001
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"Modelo guardado en: {model_path}")