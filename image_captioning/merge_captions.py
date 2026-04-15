import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\s*([-\/])\s*', r"\1", text)
    text = re.sub(r'\s*([,.;:!?%])\s*', r"\1 ", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r'\s*"\s*', '"', text)
    text = re.sub(r"\s*\'\s*", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_best_caption(norm_text: str, caption_items):
    best_caption = None
    best_ratio = 0.0

    for candidate_norm, caption in caption_items:
        ratio = SequenceMatcher(None, norm_text, candidate_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_caption = caption

    return best_caption, best_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Añade captions a cada noticia en el JSON de alertas."
    )
    parser.add_argument(
        "--alerts",
        default="selected_alert_train_examples.json",
        help="JSON con text, ner_output y sa_output",
    )
    parser.add_argument(
        "--captions",
        default="imagenes_generadas/captions.json",
        help="JSON con text y caption",
    )
    parser.add_argument(
        "--output",
        default="selected_alert_train_examples_with_captions.json",
        help="JSON de salida combinado",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Umbral de similitud para emparejar textos si no hay coincidencia exacta",
    )
    args = parser.parse_args()

    alerts_path = Path(args.alerts)
    captions_path = Path(args.captions)
    output_path = Path(args.output)

    alerts = load_json(alerts_path)
    captions = load_json(captions_path)

    caption_items = [
        (normalize_text(item["text"]), item.get("caption"))
        for item in captions
        if "text" in item
    ]
    caption_map = {norm_text: caption for norm_text, caption in caption_items}

    for alert in alerts:
        norm_alert = normalize_text(alert.get("text", ""))
        caption = caption_map.get(norm_alert)

        if caption is None:
            caption, ratio = find_best_caption(norm_alert, caption_items)
            if ratio < args.threshold:
                caption = None
            else:
                alert["_matched_caption_similarity"] = round(ratio, 3)

        alert["caption"] = caption

    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(alerts, out_file, ensure_ascii=False, indent=2)

    print(f"Generado: {output_path}")


if __name__ == "__main__":
    main()