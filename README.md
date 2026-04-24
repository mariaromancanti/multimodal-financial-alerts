# Final Project NLP + DL

Este proyecto construye un pipeline para:

1. reducir el dataset original financiero,
2. entrenar o reutilizar un modelo de `NER`,
3. entrenar o reutilizar un modelo de `sentiment analysis`,
4. entrenar o reutilizar un modelo conjunto `NER + SA`,
5. generar captions para imágenes asociadas a 20 ejemplos seleccionados,
6. generar alertas financieras combinando texto, entidades, sentimiento y caption.

## Estructura

- `datos_orginales/`: preparación y comprobación de los datos reducidos.
- `ner/`: entrenamiento, evaluación e inferencia de NER.
- `sa/`: entrenamiento, evaluación e inferencia de sentiment analysis.
- `joint/`: modelo multitarea conjunto NER + SA.
- `image_captioning/`: generación de captions para las imágenes.
- `alert/`: generación de alertas financieras.
- `data/`: datos reducidos y salidas del pipeline.

## Requisitos

Usa Python 3.10+ y crea un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pandas scikit-learn transformers datasets tqdm ollama
```

## Dependencias externas

Para la parte de captions y generación de alertas necesitas un servidor local compatible con `ollama`.

Modelos usados en el código:

- captions: `llava-phi3`
- alert generation: `qwen2.5:3b`

Ejemplo:
```bash
ollama pull llava-phi3
ollama pull qwen2.5:3b
```

## Ejecución rápida

Si solo quieres ejecutar la parte final, este repositorio ya incluye:

- `ner/models/bilstm_ner.pt`
- `sa/models/bilstm_sa.pt`
- `joint/models/bilstm_joint_ner_sa.pt`
- `selected_alert_train_examples_with_captions.json`

Con eso puedes ir directamente a `Alert Generation Combined`.

## Pipeline completo paso a paso

### 1. Preparar datos reducidos

Esto genera:

- `data/train_reduced.jsonl`
- `data/validation_reduced.jsonl`
- `data/test_reduced.jsonl`

```bash
python3 datos_orginales/PROCESING_DATOS.py
```

Si quieres comprobar que la reducción está bien:

```bash
python3 datos_orginales/comprobaciones.py
```

### 2. Entrenar o evaluar NER

El script `ner/main.py` por defecto evalúa el modelo guardado. Si ya existe `ner/models/bilstm_ner.pt`, basta con:

```bash
python3 ner/main.py
```

Si quieres entrenarlo desde cero:

```bash
cd ner
python3 -c "from main import train_main; train_main()"
cd ..
```

Esto usa:

- `data/train_reduced.jsonl`
- `data/validation_reduced.jsonl`
- vocabularios en `vocab/`
- modelo final en `ner/models/bilstm_ner.pt`

### 3. Generar datos con sentimiento

Para construir el dataset de SA con etiqueta de sentimiento:

```bash
python3 sa/build_sentiment_finbert.py
```

Esto genera:

- `sa/reduced_data_with_sentiment/train_with_sentiment.jsonl`
- `sa/reduced_data_with_sentiment/validation_with_sentiment.jsonl`
- `sa/reduced_data_with_sentiment/test_with_sentiment.jsonl`

Nota:

- el script lee desde `sa/reduced_data/`
- En el caso de que los ficheros esten en data (si se siguen estos pasos en orden no ocurrirá) `data/`, primero hay que copiarlos a `sa/reduced_data/`

### 4. Entrenar o evaluar SA

Para evaluar el modelo guardado:

```bash
cd sa
python3 main_sa.py
cd ..
```

Para entrenarlo:

```bash
cd sa
python3 -c "from main_sa import train_main; train_main()"
cd ..
```

Esto genera o usa:

- vocabulario en `sa/vocab_sa/`
- modelo en `sa/models/bilstm_sa.pt`

### 5. Entrenar el modelo conjunto NER + SA

El modelo conjunto se guarda en:

- `joint/models/bilstm_joint_ner_sa.pt`

Si el archivo no existe, el propio pipeline combinado puede entrenarlo automáticamente cuando haga falta. Si quieres forzar el entrenamiento desde código, la entrada principal es:

- [joint/train_joint.py](/home/javimendozagr/DEEP_LEARNING/Proyecto/GITHUB/final-project-nlp-dl-opositores/joint/train_joint.py:1)

El entrenamiento conjunto usa:

- vocabulario NER de `vocab/`
- etiquetas de sentimiento de `sa/vocab_sa/`
- datos con sentimiento

## Pipeline de captions

Esta parte trabaja sobre 20 ejemplos seleccionados con imagen.

### 6. Extraer los 20 ejemplos seleccionados

```bash
python3 extract_selected_alert_examples.py
```

Salida:

- `selected_alert_train_examples.json`

### 7. Generar captions para las imágenes

```bash
python3 image_captioning/generar_json.py
```

Salida:

- `imagenes_generadas/captions.json`

Este script espera:

- imágenes dentro de `imagenes_generadas/`
- lista de ejemplos en `imagenes_generadas/selected_sa_examples.txt`

### 8. Unir ejemplos y captions

```bash
python3 image_captioning/merge_captions.py
```

Salida:

- `selected_alert_train_examples_with_captions.json`

## Alert generation base

El script base es:

- [alert/alert_main.py](/home/javimendozagr/DEEP_LEARNING/Proyecto/GITHUB/final-project-nlp-dl-opositores/alert/alert_main.py:1)

Sirve para:

- construir datasets procesados para alertas,
- reutilizar NER y SA,
- generar alertas sobre validación.

Ejecución:

```bash
python3 alert/alert_main.py
```

Genera o actualiza archivos como:

- `data/alert_train_processed.json`
- `data/alert_val_processed.json`
- `data/alert_val_predictions_ollama_strict.json`

## Alert Generation Combined

El script principal para la comparativa de combinaciones es:

```bash
python3 alert/alert_main_combinaciones.py
```

Te mostrará este menú:

1. `NER solo`
2. `SA solo`
3. `Image captioning solo`
4. `NER + SA`
5. `NER + SA + Image captioning`
6. `NER + Image captioning`
7. `SA + Image captioning`

## Qué usa exactamente el script combinado

Actualmente todas las combinaciones trabajan sobre los mismos 20 ejemplos de:

- `selected_alert_train_examples_with_captions.json`

Eso significa:

- si eliges una opción sin caption, usa esos 20 ejemplos pero ignora el campo `caption`
- si eliges una opción con caption, usa ese mismo conjunto y añade `caption` al prompt

Las predicciones se guardan en `data/` con nombres como:

- `data/alert_predictions_ner_only.json`
- `data/alert_predictions_sa_only.json`
- `data/alert_predictions_caption_only.json`
- `data/alert_predictions_ner_sa.json`
- `data/alert_predictions_ner_sa_caption.json`
- `data/alert_predictions_ner_caption.json`
- `data/alert_predictions_sa_caption.json`

## Orden recomendado para ejecutar todo

Si quieres reproducir todo de principio a fin, este sería el orden más razonable:

1. `python3 datos_orginales/PROCESING_DATOS.py`
2. entrenar NER
3. generar datos con sentimiento
4. entrenar SA
5. extraer los 20 ejemplos seleccionados
6. generar captions
7. unir captions con los ejemplos seleccionados
8. ejecutar `python3 alert/alert_main_combinaciones.py`

## Orden recomendado si el repo ya trae modelos

Si solo quieres ver resultados finales:

1. asegúrate de tener `ollama` levantado
2. asegúrate de tener disponibles `llava-phi3` y `qwen2.5:3b`
3. ejecuta `python3 alert/alert_main_combinaciones.py`
4. elige la combinación que quieras

## Archivos importantes de salida

- `data/train_reduced.jsonl`
- `data/validation_reduced.jsonl`
- `data/test_reduced.jsonl`
- `ner/models/bilstm_ner.pt`
- `sa/models/bilstm_sa.pt`
- `joint/models/bilstm_joint_ner_sa.pt`
- `selected_alert_train_examples.json`
- `imagenes_generadas/captions.json`
- `selected_alert_train_examples_with_captions.json`
- `data/alert_predictions_*.json`

## Ayuda para la reproducción del proyecto

### Falta un modelo `.pt`

Entrena esa parte antes:

- NER: `cd ner && python3 -c "from main import train_main; train_main()" && cd ..`
- SA: `cd sa && python3 -c "from main_sa import train_main; train_main()" && cd ..`

### Falla la generación de captions o alertas

Comprueba:

- que `ollama` está corriendo
- que el modelo solicitado está descargado
- que el endpoint por defecto sigue siendo `http://127.0.0.1:11434`

### Faltan los 20 ejemplos con captions

Reconstrúyelos en este orden:

```bash
python3 extract_selected_alert_examples.py
python3 image_captioning/generar_json.py
python3 image_captioning/merge_captions.py
```

## Resumen mínimo

Para ejecutar la parte más importante del proyecto:

```bash
source .venv/bin/activate
python3 alert/alert_main_combinaciones.py
```

Y después elige una combinación del menú.
