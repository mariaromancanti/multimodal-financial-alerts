import json


def build_alert_dataset_from_models(
    input_json_path,
    output_json_path,
    ner_model,
    sa_model,
    ner_vocab,
    ner_idx_to_tag,
    ner_device,
    predict_ner_fn,
    predict_sa_fn
):
    """
    Lee un dataset con textos, ejecuta NER y SA sobre cada ejemplo
    y guarda un nuevo JSON listo para entrenar Alert.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alert_data = []

    for sample in data:
        text = sample["text"]

        ner_output = predict_ner_fn(
            ner_model,
            text,
            ner_vocab,
            ner_idx_to_tag,
            ner_device
        )

        sa_output = predict_sa_fn(sa_model, text)

        entities = []
        for token, tag in ner_output:
            if tag != "O":
                entities.append(f"{token} ({tag})")

        entities_text = ", ".join(entities) if entities else "None"
        sentiment_text = sa_output

        prompt = (
            f"Entities: {entities_text}. "
            f"Sentiment: {sentiment_text}. "
            f"Generate an alert."
        )

        if "alert" in sample:
            target_alert = sample["alert"]
        else:
            target_alert = ""

        alert_data.append({
            "text": text,
            "entities": entities,
            "sentiment": sentiment_text,
            "input": prompt,
            "output": target_alert
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(alert_data, f, ensure_ascii=False, indent=4)

    print(f"Dataset de alertas guardado en: {output_json_path}")