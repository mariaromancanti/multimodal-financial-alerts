### ESTE SCRIPT LO USAMOS PARA COMPROBAR QUE LA REDUCCION TANTO EN FILAS COMO EN COLUMNAS ES CORRECTA


import json
from pathlib import Path
from collections import Counter

# CONFIG
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

EXPECTED_ROWS = {
    "train_reduced.jsonl": 2000,
    "validation_reduced.jsonl": 500,
    "test_reduced.jsonl": 500,
}

BASE_DIR = Path("reduced_data")


# HELPERS
def is_valid_tag(tag: str) -> bool:
    if tag == "O":
        return True

    if "-" not in tag:
        return False

    prefix, entity = tag.split("-", 1)

    if prefix not in {"B", "I"}:
        return False

    return entity in SELECTED_ENTITIES


def analyze_file(path: Path, expected_rows: int | None = None) -> None:
    print("=" * 80)
    print(f"Comprobando: {path.name}")

    total_rows = 0
    rows_with_entities = 0
    bad_length_rows = []
    missing_columns_rows = []
    invalid_tags = []
    all_o_rows = []

    tag_counter = Counter()
    entity_counter = Counter()
    tokens_per_row = []
    selected_tags_per_row = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total_rows += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error JSON en línea {idx}: {e}")
                return

            # 1) comprobar columnas necesarias
            if "tokens" not in sample or "ner_tags" not in sample:
                missing_columns_rows.append(idx)
                continue

            tokens = sample["tokens"]
            ner_tags = sample["ner_tags"]

            # 2) comprobar tipos básicos
            if not isinstance(tokens, list) or not isinstance(ner_tags, list):
                print(f"Línea {idx}: tokens o ner_tags no son listas")
                continue

            # 3) comprobar misma longitud
            if len(tokens) != len(ner_tags):
                bad_length_rows.append(idx)

            tokens_per_row.append(len(tokens))

            # 4) comprobar tags válidas
            selected_count = 0
            for tag in ner_tags:
                if not is_valid_tag(tag):
                    invalid_tags.append((idx, tag))
                else:
                    tag_counter[tag] += 1
                    if tag != "O":
                        selected_count += 1
                        _, entity = tag.split("-", 1)
                        entity_counter[entity] += 1

            selected_tags_per_row.append(selected_count)

            # 5) comprobar filas no vacías semánticamente
            if selected_count > 0:
                rows_with_entities += 1
            else:
                all_o_rows.append(idx)

    # RESUMEN
    print(f"Filas totales: {total_rows}")

    if expected_rows is not None:
        if total_rows == expected_rows:
            print(f"Número de filas correcto: {expected_rows}")
        else:
            print(f"Número de filas incorrecto: esperado {expected_rows}, obtenido {total_rows}")

    if missing_columns_rows:
        print(f"Filas sin columnas requeridas (primeras 10): {missing_columns_rows[:10]}")
    else:
        print("Todas las filas tienen 'tokens' y 'ner_tags'")

    if bad_length_rows:
        print(f"Filas con len(tokens) != len(ner_tags) (primeras 10): {bad_length_rows[:10]}")
    else:
        print("En todas las filas len(tokens) == len(ner_tags)")

    if invalid_tags:
        print("Se encontraron etiquetas no válidas")
        print("Primeras 10:")
        for row_id, tag in invalid_tags[:10]:
            print(f"   - fila {row_id}: {tag}")
    else:
        print("Todas las etiquetas son 'O' o entidades seleccionadas con formato B-/I-")

    if all_o_rows:
        print(f"Hay filas sin ninguna entidad seleccionada (primeras 10): {all_o_rows[:10]}")
    else:
        print("No hay filas con todas las etiquetas en 'O'")

    print(f"Filas con al menos una entidad seleccionada: {rows_with_entities}/{total_rows}")

    if tokens_per_row:
        avg_len = sum(tokens_per_row) / len(tokens_per_row)
        print(f"Longitud media de secuencia: {avg_len:.2f} tokens")

    if selected_tags_per_row:
        avg_selected = sum(selected_tags_per_row) / len(selected_tags_per_row)
        print(f"Media de tags seleccionadas por fila: {avg_selected:.2f}")

    print("\nTop 10 etiquetas más frecuentes:")
    for tag, count in tag_counter.most_common(10):
        print(f"  {tag}: {count}")

    print("\nTop 10 entidades más frecuentes:")
    for ent, count in entity_counter.most_common(10):
        print(f"  {ent}: {count}")

    print("=" * 80)
    print()


def main():
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = BASE_DIR / filename
        if not path.exists():
            print(f"No existe el archivo: {path}")
            continue

        analyze_file(path, expected_rows=expected_rows)


if __name__ == "__main__":
    main()