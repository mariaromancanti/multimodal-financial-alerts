from datasets import Dataset
from transformers import Trainer, TrainingArguments

from alert.alert_preprocessing_nuestro import prepare_alert_dataset


def build_hf_dataset(encodings):
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
    num_train_epochs=1,
    train_batch_size=2,
    eval_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
):
    if getattr(alert_generator, "backend", "hf") != "hf":
        raise RuntimeError(
            "Training is only supported with the Hugging Face backend. "
            "Use do_train=False with Ollama."
        )

    train_prompts, train_alerts, train_processed = prepare_alert_dataset(
        texts=train_texts,
        ner_outputs=train_ner_outputs,
        sa_outputs=train_sa_outputs,
        use_text=use_text,
        relevant_labels=relevant_labels,
        max_entities=max_entities,
    )

    val_prompts, val_alerts, val_processed = prepare_alert_dataset(
        texts=val_texts,
        ner_outputs=val_ner_outputs,
        sa_outputs=val_sa_outputs,
        use_text=use_text,
        relevant_labels=relevant_labels,
        max_entities=max_entities,
    )

    train_encodings = alert_generator.tokenize_data(
        prompts=train_prompts,
        alerts=train_alerts,
        max_length=128,
    )
    val_encodings = alert_generator.tokenize_data(
        prompts=val_prompts,
        alerts=val_alerts,
        max_length=128,
    )

    train_dataset = build_hf_dataset(train_encodings)
    val_dataset = build_hf_dataset(val_encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=alert_generator.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    alert_generator.save_pretrained(output_dir)

    return alert_generator, trainer, train_processed, val_processed
