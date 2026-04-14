import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alert.alert_generator_nuestro import AlertGenerator
from alert.alert_predict_nuestro import predict_alert_from_outputs
from alert.alert_train_nuestro import train_alert_generator
from ner.bidireccional_modelo import BiLSTMNER
from ner.evaluate_nuestro import predict_ner
from ner.utils import load_vocabularies
from sa.evaluate_sa import predict_sentiment
from sa.sentiment_model_sa import SentimentBiLSTM
from sa.utils import load_json as load_sa_json


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def tokens_to_text(tokens):
    return " ".join(tokens)


def build_alert_dataset_from_models(
    input_json_path,
    output_json_path,
    ner_model,
    sa_model,
    ner_vocab,
    ner_idx_to_tag,
    ner_device,
    sa_token_to_idx,
    sa_idx_to_sentiment,
    sa_device,
    predict_ner_fn,
    predict_sa_fn,
):
    data = load_json_or_jsonl(input_json_path)
    processed_data = []

    for example in data:
        if "tokens" in example:
            text = tokens_to_text(example["tokens"])
        else:
            text = example["text"]

        ner_output = predict_ner_fn(
            ner_model,
            text,
            ner_vocab,
            ner_idx_to_tag,
            ner_device,
        )

        sa_output = predict_sa_fn(
            sa_model,
            text,
            sa_token_to_idx,
            sa_idx_to_sentiment,
            sa_device,
        )

        processed_data.append(
            {
                "text": text,
                "ner_output": ner_output,
                "sa_output": sa_output,
            }
        )

    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

    return processed_data


def unpack_examples(examples):
    texts = [example["text"] for example in examples]
    ner_outputs = [example["ner_output"] for example in examples]
    sa_outputs = [example["sa_output"] for example in examples]
    return texts, ner_outputs, sa_outputs


# def build_ner_model(device):
#     token_vocab_path = PROJECT_ROOT / "vocab" / "token_to_idx.json"
#     tag_vocab_path = PROJECT_ROOT / "vocab" / "tag_to_idx.json"
#     model_path = PROJECT_ROOT / "ner" / "models" / "bilstm_ner.pt"

#     ner_vocab, tag_to_idx, idx_to_tag = load_vocabularies(
#         str(token_vocab_path),
#         str(tag_vocab_path),
#     )

#     if "<OOV>" not in ner_vocab and "<UNK>" in ner_vocab:
#         ner_vocab["<OOV>"] = ner_vocab["<UNK>"]

#     model = BiLSTMNER(
#         vocab_size=len(ner_vocab),
#         embedding_dim=100,
#         hidden_dim=128,
#         tagset_size=len(tag_to_idx),
#         dropout=0.2,
#     ).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     return model, ner_vocab, idx_to_tag

def build_ner_model(device):
    import json
    import torch
    from ner.bidireccional_modelo import BiLSTMNER

    vocab_path = "vocab/token_to_idx.json"
    tag_path = "vocab/tag_to_idx.json"
    model_path = "ner/models/bilstm_ner.pt"

    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_idx = json.load(f)

    with open(tag_path, "r", encoding="utf-8") as f:
        tag_to_idx = json.load(f)

    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

    model = BiLSTMNER(
        vocab_size=len(token_to_idx),
        embedding_dim=100,
        hidden_dim=128,
        tagset_size=len(tag_to_idx),
        padding_idx=0,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if "<OOV>" not in token_to_idx and "<UNK>" in token_to_idx:
        token_to_idx["<OOV>"] = token_to_idx["<UNK>"]

    return model, token_to_idx, idx_to_tag

def build_sa_model(device):
    token_vocab_path = PROJECT_ROOT / "sa" / "vocab_sa" / "token_to_idx_sa.json"
    sentiment_vocab_path = PROJECT_ROOT / "sa" / "vocab_sa" / "sentiment_to_idx.json"
    model_path = PROJECT_ROOT / "sa" / "models" / "bilstm_sa.pt"

    token_to_idx = load_sa_json(token_vocab_path)
    sentiment_to_idx = load_sa_json(sentiment_vocab_path)
    idx_to_sentiment = {idx: label for label, idx in sentiment_to_idx.items()}

    model = SentimentBiLSTM(
        vocab_size=len(token_to_idx),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=len(sentiment_to_idx),
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, token_to_idx, idx_to_sentiment


def main(
    train_json_path=None,
    val_json_path=None,
    processed_train_json_path=None,
    processed_val_json_path=None,
    alert_model_name="distilgpt2",
    alert_device="cpu",
    do_train=False,
):
    ner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sa_device = ner_device

    if train_json_path is None:
        train_json_path = str(PROJECT_ROOT / "data" / "train_reduced.jsonl")
    if val_json_path is None:
        val_json_path = str(PROJECT_ROOT / "data" / "validation_reduced.jsonl")
    if processed_train_json_path is None:
        processed_train_json_path = str(PROJECT_ROOT / "data" / "alert_train_processed.json")
    if processed_val_json_path is None:
        processed_val_json_path = str(PROJECT_ROOT / "data" / "alert_val_processed.json")

    ner_model, ner_vocab, ner_idx_to_tag = build_ner_model(ner_device)
    sa_model, sa_token_to_idx, sa_idx_to_sentiment = build_sa_model(sa_device)

    print("Building processed train dataset...")
    train_examples = build_alert_dataset_from_models(
        input_json_path=train_json_path,
        output_json_path=processed_train_json_path,
        ner_model=ner_model,
        sa_model=sa_model,
        ner_vocab=ner_vocab,
        ner_idx_to_tag=ner_idx_to_tag,
        ner_device=ner_device,
        sa_token_to_idx=sa_token_to_idx,
        sa_idx_to_sentiment=sa_idx_to_sentiment,
        sa_device=sa_device,
        predict_ner_fn=predict_ner,
        predict_sa_fn=predict_sentiment,
    )

    print("Building processed validation dataset...")
    val_examples = build_alert_dataset_from_models(
        input_json_path=val_json_path,
        output_json_path=processed_val_json_path,
        ner_model=ner_model,
        sa_model=sa_model,
        ner_vocab=ner_vocab,
        ner_idx_to_tag=ner_idx_to_tag,
        ner_device=ner_device,
        sa_token_to_idx=sa_token_to_idx,
        sa_idx_to_sentiment=sa_idx_to_sentiment,
        sa_device=sa_device,
        predict_ner_fn=predict_ner,
        predict_sa_fn=predict_sentiment,
    )

    print(f"Processed train examples: {len(train_examples)}")
    print(f"Processed validation examples: {len(val_examples)}")

    train_texts, train_ner_outputs, train_sa_outputs = unpack_examples(train_examples)
    val_texts, val_ner_outputs, val_sa_outputs = unpack_examples(val_examples)

    alert_generator = AlertGenerator(
        model_name=alert_model_name,
        device=alert_device,
    )

    trainer = None
    train_processed = None
    val_processed = None

    if do_train:
        print("Training alert generator...")
        alert_generator, trainer, train_processed, val_processed = train_alert_generator(
            alert_generator=alert_generator,
            train_texts=train_texts,
            train_ner_outputs=train_ner_outputs,
            train_sa_outputs=train_sa_outputs,
            val_texts=val_texts,
            val_ner_outputs=val_ner_outputs,
            val_sa_outputs=val_sa_outputs,
            use_text=False,
            max_entities=3,
        )
        print("Alert generator training finished.")

    if len(val_examples) > 0:
        sample_example = val_examples[0]
        prediction = predict_alert_from_outputs(
            alert_generator=alert_generator,
            text=sample_example["text"],
            ner_output=sample_example["ner_output"],
            sa_output=sample_example["sa_output"],
            use_text=False,
            max_entities=3,
        )

        print("\n--- SAMPLE PREDICTION ---")
        print("TEXT:")
        print(prediction["text"])
        print("\nPROMPT:")
        print(prediction["prompt"])
        print("\nTARGET ALERT:")
        print(prediction["target_alert"])
        print("\nGENERATED ALERT:")
        print(prediction["generated_alert"])

    return alert_generator, trainer, train_examples, val_examples, train_processed, val_processed


if __name__ == "__main__":
    main(do_train=False)
