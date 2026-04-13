import json

from alert_generator_nuestro import AlertGenerator
from alert_train_nuestro import train_alert_generator
from alert_predict_nuestro import predict_alert_from_outputs


def load_json(json_path):
    """
    Carga un archivo JSON y devuelve su contenido.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_text(tokens):
    """
    Convierte una lista de tokens en texto.
    """
    return " ".join(tokens)


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
    Lee un dataset con tokens, ejecuta NER y SA sobre cada ejemplo
    y guarda un nuevo JSON listo para alert generation.
    """
    data = load_json(input_json_path)
    processed_data = []

    for example in data:
        tokens = example["tokens"]
        text = tokens_to_text(tokens)

        ner_output = predict_ner_fn(
            ner_model,
            text,
            ner_vocab,
            ner_idx_to_tag,
            ner_device
        )

        sa_output = predict_sa_fn(
            sa_model,
            text
        )

        processed_data.append({
            "text": text,
            "ner_output": ner_output,
            "sa_output": sa_output
        })

    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

    return processed_data


def unpack_examples(examples):
    """
    Separa los ejemplos procesados en listas:
    texts, ner_outputs, sa_outputs
    """
    texts = [example["text"] for example in examples]
    ner_outputs = [example["ner_output"] for example in examples]
    sa_outputs = [example["sa_output"] for example in examples]

    return texts, ner_outputs, sa_outputs


def main(
    train_json_path,
    val_json_path,
    ner_model,
    sa_model,
    ner_vocab,
    ner_idx_to_tag,
    ner_device,
    predict_ner_fn,
    predict_sa_fn,
    processed_train_json_path="data/alert_train_processed.json",
    processed_val_json_path="data/alert_val_processed.json",
    alert_model_name="distilgpt2",
    alert_device="cpu"
):
    # =========================
    # 1. CONSTRUIR DATASET PARA ALERT GENERATION
    # =========================
    print("Building processed train dataset...")
    train_examples = build_alert_dataset_from_models(
        input_json_path=train_json_path,
        output_json_path=processed_train_json_path,
        ner_model=ner_model,
        sa_model=sa_model,
        ner_vocab=ner_vocab,
        ner_idx_to_tag=ner_idx_to_tag,
        ner_device=ner_device,
        predict_ner_fn=predict_ner_fn,
        predict_sa_fn=predict_sa_fn
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
        predict_ner_fn=predict_ner_fn,
        predict_sa_fn=predict_sa_fn
    )

    print(f"Processed train examples: {len(train_examples)}")
    print(f"Processed validation examples: {len(val_examples)}")

    # =========================
    # 2. SEPARAR EN LISTAS
    # =========================
    train_texts, train_ner_outputs, train_sa_outputs = unpack_examples(train_examples)
    val_texts, val_ner_outputs, val_sa_outputs = unpack_examples(val_examples)

    # =========================
    # 3. INICIALIZAR ALERT GENERATOR
    # =========================
    alert_generator = AlertGenerator(
        model_name=alert_model_name,
        device=alert_device
    )

    # =========================
    # 4. ENTRENAR ALERT GENERATOR
    # =========================
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
        max_entities=3
    )

    print("Alert generator training finished.")

    # =========================
    # 5. EJEMPLO DE PREDICCIÓN
    # =========================
    if len(val_examples) > 0:
        sample_example = val_examples[0]

        prediction = predict_alert_from_outputs(
            alert_generator=alert_generator,
            text=sample_example["text"],
            ner_output=sample_example["ner_output"],
            sa_output=sample_example["sa_output"],
            use_text=False,
            max_entities=3
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

    return alert_generator, trainer, train_examples, val_examples


if __name__ == "__main__":
    from ner.evaluate_nuestro import predict_ner
    from sa.evaluate_sa import predict_sentiment

    # Aquí tienes que sustituir esto por tu carga real de modelos
    ner_model = None
    sa_model = None
    ner_vocab = None
    ner_idx_to_tag = None
    ner_device = "cpu"



    main(
        train_json_path="data/train.json",
        val_json_path="data/val.json",
        ner_model=ner_model,
        sa_model=sa_model,
        ner_vocab=ner_vocab,
        ner_idx_to_tag=ner_idx_to_tag,
        ner_device=ner_device,
        predict_ner_fn=predict_ner,
        predict_sa_fn=predict_sentiment,
        processed_train_json_path="data/alert_train_processed.json",
        processed_val_json_path="data/alert_val_processed.json",
        alert_model_name="distilgpt2",
        alert_device="cpu"
    )