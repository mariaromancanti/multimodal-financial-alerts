from alert_preprocessing_nuestro import prepare_alert_example


def predict_alert(alert_generator, prompt):
    """
    Genera una alerta a partir de un prompt ya construido.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("El prompt debe ser un string no vacío.")

    return alert_generator.generate_alert(prompt.strip())


def predict_alert_from_outputs(
    alert_generator,
    text,
    ner_output,
    sa_output,
    use_text=False,
    relevant_labels=None,
    max_entities=3
):
    """
    Genera una alerta usando el texto y las salidas ya obtenidas
    de NER y SA.
    """
    example = prepare_alert_example(
        text=text,
        ner_output=ner_output,
        sa_output=sa_output,
        use_text=use_text,
        relevant_labels=relevant_labels,
        max_entities=max_entities
    )

    generated_alert = alert_generator.generate_alert(example["prompt"])

    return {
        "text": text,
        "entities": example["entities"],
        "sentiment": example["sentiment"],
        "prompt": example["prompt"],
        "target_alert": example["alert"],
        "generated_alert": generated_alert
    }