from transformers import Trainer, TrainingArguments
from datasets import Dataset

from alert_preprocessing_nuestro import prepare_alert_dataset 


def build_hf_dataset(encodings):
    """
    Convierte los tensores tokenizados en un Dataset de Hugging Face.
    """
    dataset_dict = {
        "input_ids": encodings["input_ids"].cpu().tolist(),
        "attention_mask": encodings["attention_mask"].cpu().tolist(),
    }

    if "labels" in encodings:
        dataset_dict["labels"] = encodings["labels"].cpu().tolist()

    return Dataset.from_dict(dataset_dict)


def train_alert_generator(
    alert_generator,
    train_texts,
    train_ner_outputs,
    train_sa_outputs,
    val_texts,
    val_ner_outputs,
    val_sa_outputs,
    use_text=False,
    relevant_labels=None,
    max_entities=3,
    output_dir="./alert_generator",
    num_train_epochs=3,
    train_batch_size=4,
    eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01
):
    """
    Entrena el generador de alertas a partir de las salidas de NER + SA.
    """

    # Preparar prompts y alerts del conjunto de entrenamiento
    train_prompts, train_alerts, train_processed = prepare_alert_dataset(
        texts=train_texts,
        ner_outputs=train_ner_outputs,
        sa_outputs=train_sa_outputs,
        use_text=use_text,
        relevant_labels=relevant_labels,
        max_entities=max_entities
    )

    # Preparar prompts y alerts del conjunto de validación
    val_prompts, val_alerts, val_processed = prepare_alert_dataset(
        texts=val_texts,
        ner_outputs=val_ner_outputs,
        sa_outputs=val_sa_outputs,
        use_text=use_text,
        relevant_labels=relevant_labels,
        max_entities=max_entities
    )

    # Tokenizar
    train_encodings = alert_generator.tokenize_data(
        prompts=train_prompts,
        alerts=train_alerts,
        max_length=128
    )

    val_encodings = alert_generator.tokenize_data(
        prompts=val_prompts,
        alerts=val_alerts,
        max_length=128
    )

    # Convertir a Dataset HF
    train_dataset = build_hf_dataset(train_encodings)
    val_dataset = build_hf_dataset(val_encodings)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=False,   # poner True solo si entrenas en GPU compatible
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=alert_generator.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Entrenamiento
    trainer.train()

    return alert_generator, trainer, train_processed, val_processed