### codigo para el procesamiento de los datos:

# QUEREMOS FILTRAR TANTO EN FILAS: QUEDARNOS CON UNAS 2000 QUE CONTENGAN UNA CANTIDAD ELEVADA DE
#  NER TAGS COMO EN COLUMNAS, QUEDARNOS SOLO CON UNAS 12 NER TAGS, LAS MAS IMPORTANTES
# AQUI VAMOS A GENERAR:
# train_reduced.jsonl
# validation_reduced.jsonl
# test_reduced.jsonl


import pandas as pd
from pathlib import Path

SELECTED_ENTITIES = {
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "IncomeLossFromEquityMethodInvestments",
    "OperatingLossCarryforwards",
    "RestructuringCharges",
    "AssetImpairmentCharges",
    "LongTermDebt",
    "InterestExpense",
    "IncomeTaxExpenseBenefit",
    "EffectiveIncomeTaxRateContinuingOperations",
    "DebtInstrumentInterestRateStatedPercentage",
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "DebtInstrumentBasisSpreadOnVariableRate",
    "DebtInstrumentFaceAmount",
    "AllocatedShareBasedCompensationExpense",
    "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount",
}

INPUT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}

TARGET_ROWS = {
    "train": 2000,
    "validation": 500,
    "test": 500,
}

OUTPUT_DIR = Path("reduced_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_entity_name(entity: str) -> str:
    return entity.rstrip("0123456789")


def filter_tags(tags):
    new_tags = []
    for tag in tags:
        if tag == "O":
            new_tags.append("O")
            continue

        if "-" not in tag:
            new_tags.append("O")
            continue

        prefix, entity = tag.split("-", 1)
        entity_base = normalize_entity_name(entity)

        if entity in SELECTED_ENTITIES:
            new_tags.append(f"{prefix}-{entity}")
        elif entity_base in SELECTED_ENTITIES:
            new_tags.append(f"{prefix}-{entity_base}")
        else:
            new_tags.append("O")

    return new_tags


def count_selected_tokens(tags):
    return sum(tag != "O" for tag in tags)


def count_selected_spans(tags):
    return sum(tag.startswith("B-") for tag in tags)


def process_split(split_name, input_path, output_rows, chunksize=50000):
    selected_chunks = []

    for chunk in pd.read_json(input_path, lines=True, chunksize=chunksize):
        keep_cols = [c for c in ["tokens", "ner_tags"] if c in chunk.columns]
        chunk = chunk[keep_cols].copy()

        chunk["ner_tags"] = chunk["ner_tags"].apply(filter_tags)
        chunk["selected_token_count"] = chunk["ner_tags"].apply(count_selected_tokens)
        chunk["selected_span_count"] = chunk["ner_tags"].apply(count_selected_spans)
        chunk["seq_len"] = chunk["tokens"].apply(len)

        chunk = chunk[chunk["selected_token_count"] > 0]

        if not chunk.empty:
            selected_chunks.append(chunk)

    if not selected_chunks:
        print(f"[{split_name}] No se encontraron filas válidas")
        return

    df = pd.concat(selected_chunks, ignore_index=True)

    df = df.sort_values(
        by=["selected_token_count", "selected_span_count", "seq_len"],
        ascending=[False, False, True]
    ).head(output_rows)

    df = df.drop(columns=["selected_token_count", "selected_span_count", "seq_len"])

    output_path = OUTPUT_DIR / f"{split_name}_reduced.jsonl"
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"[{split_name}] guardado en {output_path} con {len(df)} filas")


if __name__ == "__main__":
    for split_name, file_path in INPUT_FILES.items():
        process_split(split_name, file_path, TARGET_ROWS[split_name])