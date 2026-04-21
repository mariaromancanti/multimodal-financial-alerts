import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alert.alert_generator import AlertGenerator
from alert.alert_main import (
    build_ner_model,
    build_sa_model,
    load_json_or_jsonl,
    tokens_to_text,
)
from alert.alert_preprocessing import (
    bio_to_entities,
    build_combination_prompt,
    extract_sentiment_label,
    filter_relevant_entities,
    normalize_sentiment,
)
from joint.train_joint import build_joint_model, ensure_joint_model, predict_joint
from ner.evaluate import predict_ner
from ner.utils import load_vocabularies
from sa.evaluate_sa import predict_sentiment
from sa.utils import load_json


COMBINATIONS = {
    "1": {
        "name": "NER solo",
        "slug": "ner_only",
        "include_ner": True,
        "include_sa": False,
        "include_caption": False,
        "dataset": "captions",
    },
    "2": {
        "name": "SA solo",
        "slug": "sa_only",
        "include_ner": False,
        "include_sa": True,
        "include_caption": False,
        "dataset": "captions",
    },
    "3": {
        "name": "Image captioning solo",
        "slug": "caption_only",
        "include_ner": False,
        "include_sa": False,
        "include_caption": True,
        "dataset": "captions",
    },
    "4": {
        "name": "NER + SA",
        "slug": "ner_sa",
        "include_ner": True,
        "include_sa": True,
        "include_caption": False,
        "dataset": "captions",
    },
    "5": {
        "name": "NER + SA + Image captioning",
        "slug": "ner_sa_caption",
        "include_ner": True,
        "include_sa": True,
        "include_caption": True,
        "dataset": "captions",
    },
    "6": {
        "name": "NER + Image captioning",
        "slug": "ner_caption",
        "include_ner": True,
        "include_sa": False,
        "include_caption": True,
        "dataset": "captions",
    },
    "7": {
        "name": "SA + Image captioning",
        "slug": "sa_caption",
        "include_ner": False,
        "include_sa": True,
        "include_caption": True,
        "dataset": "captions",
    },
}

VALIDATION_LIMIT = 20


def prompt_user_for_combination():
    print("Selecciona la combinacion para generar alertas:")
    for key, config in COMBINATIONS.items():
        print(f"{key}. {config['name']}")

    while True:
        choice = input("Opcion: ").strip()
        if choice in COMBINATIONS:
            return COMBINATIONS[choice]

        print("Opcion no valida. Elige uno de los numeros mostrados.")


def load_validation_examples():
    validation_path = PROJECT_ROOT / "data" / "validation_reduced.jsonl"
    return load_json_or_jsonl(str(validation_path))[:VALIDATION_LIMIT]


def load_caption_examples():
    captions_path = PROJECT_ROOT / "selected_alert_train_examples_with_captions.json"
    with captions_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_models_if_needed(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"device": device}

    if config.get("joint_training"):
        joint_model_path = ensure_joint_model(device=device, force_retrain=False)
        token_to_idx, tag_to_idx, idx_to_tag = load_vocabularies(
            PROJECT_ROOT / "vocab" / "token_to_idx.json",
            PROJECT_ROOT / "vocab" / "tag_to_idx.json",
        )
        sentiment_to_idx = load_json(
            PROJECT_ROOT / "sa" / "vocab_sa" / "sentiment_to_idx.json"
        )
        idx_to_sentiment = {
            idx: label for label, idx in sentiment_to_idx.items()
        }

        joint_model = build_joint_model(
            vocab_size=len(token_to_idx),
            ner_tagset_size=len(tag_to_idx),
            sentiment_num_classes=len(sentiment_to_idx),
            device=device,
        )
        joint_model.load_state_dict(torch.load(joint_model_path, map_location=device))
        joint_model.eval()

        models["joint_model"] = joint_model
        models["joint_token_to_idx"] = token_to_idx
        models["joint_idx_to_tag"] = idx_to_tag
        models["joint_idx_to_sentiment"] = idx_to_sentiment
        return models

    if config["dataset"] == "validation" and config["include_ner"]:
        ner_model, ner_vocab, ner_idx_to_tag = build_ner_model(device)
        models["ner_model"] = ner_model
        models["ner_vocab"] = ner_vocab
        models["ner_idx_to_tag"] = ner_idx_to_tag

    if config["dataset"] == "validation" and config["include_sa"]:
        sa_model, sa_token_to_idx, sa_idx_to_sentiment = build_sa_model(device)
        models["sa_model"] = sa_model
        models["sa_token_to_idx"] = sa_token_to_idx
        models["sa_idx_to_sentiment"] = sa_idx_to_sentiment

    return models


def get_text_from_example(example):
    if "text" in example:
        return example["text"]
    return tokens_to_text(example["tokens"])


def get_entities(ner_output):
    if ner_output is None:
        return []
    return filter_relevant_entities(bio_to_entities(ner_output))


def get_sentiment(sa_output):
    if sa_output is None:
        return None
    return normalize_sentiment(extract_sentiment_label(sa_output))


def build_prompt_for_example(config, text, ner_output=None, sa_output=None, caption=None):
    entities = get_entities(ner_output) if config["include_ner"] else []
    sentiment = get_sentiment(sa_output) if config["include_sa"] else None

    prompt = build_combination_prompt(
        entities=entities,
        sentiment=sentiment,
        caption=caption,
        text=text,
        include_ner=config["include_ner"],
        include_sa=config["include_sa"],
        include_caption=config["include_caption"],
        include_text=True,
    )

    return {
        "text": text,
        "entities": entities,
        "sentiment": sentiment,
        "caption": caption if config["include_caption"] else None,
        "prompt": prompt,
    }


def save_predictions(predictions, output_path):
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def generate_from_validation(config, alert_generator, output_path):
    examples = load_validation_examples()
    models = build_models_if_needed(config)
    predictions = []
    total_examples = len(examples)

    print(f"Generating alerts for {total_examples} validation examples...")

    for index, example in enumerate(examples, start=1):
        if index == 1 or index % 10 == 0 or index == total_examples:
            print(f"Generating validation alert {index}/{total_examples}...")

        text = get_text_from_example(example)
        ner_output = None
        sa_output = None

        if config.get("joint_training"):
            ner_output, sa_output = predict_joint(
                models["joint_model"],
                text,
                models["joint_token_to_idx"],
                models["joint_idx_to_tag"],
                models["joint_idx_to_sentiment"],
                models["device"],
            )
        elif config["include_ner"]:
            ner_output = predict_ner(
                models["ner_model"],
                text,
                models["ner_vocab"],
                models["ner_idx_to_tag"],
                models["device"],
            )

        if config["include_sa"]:
            sa_output = predict_sentiment(
                models["sa_model"],
                text,
                models["sa_token_to_idx"],
                models["sa_idx_to_sentiment"],
                models["device"],
            )

        prepared = build_prompt_for_example(
            config=config,
            text=text,
            ner_output=ner_output,
            sa_output=sa_output,
        )

        generated_alert = alert_generator.generate_alert(prepared["prompt"])
        predictions.append(
            {
                "source_dataset": "validation_reduced",
                "text": prepared["text"],
                "entities": prepared["entities"],
                "sentiment": prepared["sentiment"],
                "caption": prepared["caption"],
                "prompt": prepared["prompt"],
                "generated_alert": generated_alert,
            }
        )

        if index % 10 == 0 or index == total_examples:
            save_predictions(predictions, output_path)

    return predictions


def generate_from_captions(config, alert_generator, output_path):
    examples = load_caption_examples()
    predictions = []
    total_examples = len(examples)

    print(f"Generating alerts for {total_examples} caption examples...")

    for index, example in enumerate(examples, start=1):
        if index == 1 or index % 5 == 0 or index == total_examples:
            print(f"Generating caption alert {index}/{total_examples}...")

        text = get_text_from_example(example)
        prepared = build_prompt_for_example(
            config=config,
            text=text,
            ner_output=example.get("ner_output"),
            sa_output=example.get("sa_output"),
            caption=example.get("caption"),
        )

        generated_alert = alert_generator.generate_alert(prepared["prompt"])
        predictions.append(
            {
                "source_dataset": "selected_alert_train_examples_with_captions",
                "text": prepared["text"],
                "entities": prepared["entities"],
                "sentiment": prepared["sentiment"],
                "caption": prepared["caption"],
                "prompt": prepared["prompt"],
                "generated_alert": generated_alert,
            }
        )

        if index % 5 == 0 or index == total_examples:
            save_predictions(predictions, output_path)

    return predictions


def main(
    alert_model_name="qwen2.5:3b",
    alert_device="cpu",
    alert_backend="ollama",
    ollama_url="http://127.0.0.1:11434/api/generate",
):
    config = prompt_user_for_combination()
    do_train = False

    print(f"Selected combination: {config['name']}")
    print(f"Alert generator training enabled: {do_train}")
    if config.get("joint_training"):
        print("Joint NER+SA training is enabled for this combination.")

    alert_generator = AlertGenerator(
        model_name=alert_model_name,
        device=alert_device,
        backend=alert_backend,
        ollama_url=ollama_url,
    )

    if alert_backend == "ollama":
        print(f"Validating configured model '{alert_model_name}'...")
        alert_generator.validate_ollama_configuration()

    output_path = PROJECT_ROOT / "data" / f"alert_predictions_{config['slug']}.json"
    print(f"Predictions will be saved to: {output_path}")

    if config["dataset"] == "validation":
        predictions = generate_from_validation(config, alert_generator, output_path)
    else:
        predictions = generate_from_captions(config, alert_generator, output_path)

    if predictions:
        sample = predictions[0]
        print("\n--- SAMPLE PREDICTION ---")
        print("PROMPT:")
        print(sample["prompt"])
        print("\nGENERATED ALERT:")
        print(sample["generated_alert"])


if __name__ == "__main__":
    main()
