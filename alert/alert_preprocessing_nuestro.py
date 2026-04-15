from typing import List, Dict, Tuple, Optional


# Etiquetas financieras más relevantes para generar alertas
DEFAULT_RELEVANT_LABELS = {
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "InterestExpense",
    "LongTermDebt",
    "DebtInstrumentFaceAmount",
    "DebtInstrumentInterestRateStatedPercentage",
    "DebtInstrumentBasisSpreadOnVariableRate",
    "AssetImpairmentCharges",
    "RestructuringCharges",
    "IncomeTaxExpenseBenefit",
    "EffectiveIncomeTaxRateContinuingOperations",
    "IncomeLossFromEquityMethodInvestments",
    "OperatingLossCarryforwards",
    "AllocatedShareBasedCompensationExpense",
    "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount",
    "LineOfCreditFacilityMaximumBorrowingCapacity"
}


def clean_text(text: Optional[str]) -> Optional[str]:
    """
    Limpia espacios y saltos de línea innecesarios.
    """
    if text is None:
        return None
    return " ".join(text.strip().split())


def extract_sentiment_label(sa_output) -> Optional[str]:
    """
    Extrae la etiqueta de sentimiento desde la salida del modelo SA.

    Casos admitidos:
    - "positive"
    - ("positive", {...})
    """
    if sa_output is None:
        return None

    if isinstance(sa_output, tuple):
        label = sa_output[0]
    else:
        label = sa_output

    if not isinstance(label, str):
        raise ValueError("La salida de SA debe contener una etiqueta de tipo string.")

    return label.strip().lower()


def normalize_sentiment(sentiment: Optional[str]) -> Optional[str]:
    """
    Normaliza y valida la etiqueta de sentimiento.
    """
    if sentiment is None:
        return None

    sentiment = sentiment.strip().lower()
    valid_sentiments = {"positive", "negative", "neutral"}

    if sentiment not in valid_sentiments:
        raise ValueError(f"Sentimiento no válido: {sentiment}")

    return sentiment


def bio_to_entities(ner_output: Optional[List[Tuple[str, str]]]) -> List[Dict[str, str]]:
    """
    Convierte la salida del NER [(token, tag), ...] en entidades completas.

    Ejemplo entrada:
        [("Interest", "B-InterestExpense"), ("expense", "I-InterestExpense"), ("rose", "O")]

    Ejemplo salida:
        [{"text": "Interest expense", "label": "InterestExpense"}]
    """
    if ner_output is None:
        return []

    entities = []
    current_tokens = []
    current_label = None

    for token, tag in ner_output:
        if tag == "O":
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label
                })
                current_tokens = []
                current_label = None
            continue

        if tag.startswith("B-"):
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label
                })

            current_label = tag[2:]
            current_tokens = [token]

        elif tag.startswith("I-"):
            label = tag[2:]

            if current_tokens and current_label == label:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "label": current_label
                    })

                current_label = label
                current_tokens = [token]

        else:
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_label
                })
                current_tokens = []
                current_label = None

    if current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "label": current_label
        })

    return entities


def filter_relevant_entities(
    entities: List[Dict[str, str]],
    relevant_labels: Optional[set] = None,
    max_entities: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Filtra entidades para quedarte solo con las más útiles en alert generation.
    """
    if relevant_labels is None:
        relevant_labels = DEFAULT_RELEVANT_LABELS

    filtered = [ent for ent in entities if ent["label"] in relevant_labels]
    if max_entities is None or max_entities <= 0:
        return filtered

    return filtered[:max_entities]


def format_entities_for_prompt(entities: List[Dict[str, str]]) -> str:
    """
    Convierte las entidades a una cadena legible para el prompt.
    """
    if not entities:
        return "No relevant financial entities detected."

    return ", ".join(
        f"{ent['text']} ({ent['label']})"
        for ent in entities
    )


def build_prompt(
    entities: List[Dict[str, str]],
    sentiment: str,
    text: Optional[str] = None,
    use_text: bool = False
) -> str:
    """
    Construye el prompt para el generador.
    Por defecto usa NER + SA, que es el caso base del proyecto.
    """
    lines = [
        "Generate a clear financial alert in English from the following information."
    ]

    if use_text and text:
        lines.append(f"Article: {text}")

    lines.append(f"Entities: {format_entities_for_prompt(entities)}")
    lines.append(f"Sentiment: {sentiment}")

    return "\n".join(lines)


def build_target_alert(
    entities: List[Dict[str, str]],
    sentiment: str
) -> str:
    """
    Construye una alerta objetivo simple para entrenamiento supervisado.
    """
    if not entities:
        if sentiment == "negative":
            return "Financial alert: negative sentiment detected in the article."
        if sentiment == "positive":
            return "Financial alert: positive sentiment detected in the article."
        return "Financial alert: neutral financial context detected in the article."

    entity_names = [ent["text"] for ent in entities]
    entity_text = ", ".join(entity_names)

    if sentiment == "negative":
        return f"Financial risk alert: negative context detected around {entity_text}."
    elif sentiment == "positive":
        return f"Positive financial alert: positive context detected around {entity_text}."
    else:
        return f"Informational financial alert: neutral context detected around {entity_text}."


def prepare_alert_example(
    text: str,
    ner_output: List[Tuple[str, str]],
    sa_output,
    use_text: bool = False,
    relevant_labels: Optional[set] = None,
    max_entities: Optional[int] = None
) -> Dict[str, object]:
    """
    Prepara un ejemplo completo para alert generation.

    Devuelve:
    - clean_text
    - entities
    - sentiment
    - prompt
    - alert
    """
    cleaned_text = clean_text(text)

    raw_sentiment = extract_sentiment_label(sa_output)
    sentiment = normalize_sentiment(raw_sentiment)

    entities = bio_to_entities(ner_output)
    entities = filter_relevant_entities(
        entities,
        relevant_labels=relevant_labels,
        max_entities=max_entities
    )

    prompt = build_prompt(
        entities=entities,
        sentiment=sentiment,
        text=cleaned_text,
        use_text=use_text
    )

    alert = build_target_alert(
        entities=entities,
        sentiment=sentiment
    )

    return {
        "clean_text": cleaned_text,
        "entities": entities,
        "sentiment": sentiment,
        "prompt": prompt,
        "alert": alert
    }


def prepare_alert_dataset(
    texts: List[str],
    ner_outputs: List[List[Tuple[str, str]]],
    sa_outputs: List,
    use_text: bool = False,
    relevant_labels: Optional[set] = None,
    max_entities: Optional[int] = None
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    """
    Prepara un dataset completo para entrenamiento del generador.

    Devuelve:
    - prompts
    - alerts
    - processed_examples
    """
    prompts = []
    alerts = []
    processed_examples = []

    for text, ner_output, sa_output in zip(texts, ner_outputs, sa_outputs):
        example = prepare_alert_example(
            text=text,
            ner_output=ner_output,
            sa_output=sa_output,
            use_text=use_text,
            relevant_labels=relevant_labels,
            max_entities=max_entities
        )

        prompts.append(example["prompt"])
        alerts.append(example["alert"])
        processed_examples.append(example)

    return prompts, alerts, processed_examples
