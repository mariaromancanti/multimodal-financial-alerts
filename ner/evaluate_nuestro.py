import torch


def predict_ner(model, text, vocab, idx_to_tag, device):
    model.eval()

    tokens = text.split()
    encoded = [vocab.get(tok, vocab["<OOV>"]) for tok in tokens]
    batch = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(batch)
        pred_ids = scores.argmax(dim=-1)[0]

    return [(tok, idx_to_tag[tag_id.item()]) for tok, tag_id in zip(tokens, pred_ids)]