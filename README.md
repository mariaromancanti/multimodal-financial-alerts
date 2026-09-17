# Multimodal Financial Alert Generation

A collaborative **Natural Language Processing and Deep Learning** project that combines **Named Entity Recognition (NER)**, **sentiment analysis**, **multitask learning**, **image captioning**, and **local Large Language Models** to generate contextual financial alerts.

The project implements an end-to-end multimodal pipeline capable of processing financial text and associated images, extracting relevant information from each modality, and combining the results to generate richer financial alerts.

---

## Overview

The pipeline integrates several AI components:

1. Financial dataset preprocessing
2. Named Entity Recognition (NER)
3. Sentiment Analysis (SA)
4. Joint NER + Sentiment Analysis model
5. Image caption generation
6. Multimodal information fusion
7. LLM-based financial alert generation

The final stage allows different combinations of textual and visual information to be compared when generating financial alerts.

---

## Key Features

- Financial text preprocessing
- Named Entity Recognition
- Sentiment Analysis
- Joint NER + Sentiment multitask model
- Image caption generation
- Multimodal text + image processing
- Local LLM inference using Ollama
- Financial alert generation
- Comparison of different information combinations
- Reusable trained model checkpoints

---

## Technology Stack

### Deep Learning
- Python
- PyTorch
- BiLSTM architectures
- Multitask learning

### Natural Language Processing
- Named Entity Recognition
- Sentiment Analysis
- Transformers
- FinBERT-based sentiment labeling

### Multimodal AI
- Image Captioning
- Text and image information fusion

### Generative AI
- Ollama
- LLaVA-Phi3
- Qwen 2.5

### Data & Machine Learning
- pandas
- scikit-learn
- Hugging Face Datasets
- tqdm

---

## System Architecture

```text
                     Financial Text
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
     Named Entity                  Sentiment
     Recognition                   Analysis
        (NER)                         (SA)
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
                    Joint NER + SA
                           │
                           │
Financial Image            │
      │                    │
      ▼                    │
Image Captioning           │
      │                    │
      └──────────┬─────────┘
                 │
                 ▼
        Multimodal Context
                 │
                 ▼
          Local LLM / Ollama
                 │
                 ▼
          Financial Alert
```

---

## Repository Structure

```text
.
├── datos_orginales/
│   ├── PROCESING_DATOS.py
│   └── comprobaciones.py
│
├── ner/
│   ├── main.py
│   ├── models/
│   └── ...
│
├── sa/
│   ├── main_sa.py
│   ├── build_sentiment_finbert.py
│   ├── models/
│   └── ...
│
├── joint/
│   ├── train_joint.py
│   ├── models/
│   └── ...
│
├── image_captioning/
│   ├── generar_json.py
│   ├── merge_captions.py
│   └── ...
│
├── alert/
│   ├── alert_main.py
│   ├── alert_main_combinaciones.py
│   └── ...
│
├── data/
├── imagenes_generadas/
├── extract_selected_alert_examples.py
├── selected_alert_train_examples_with_captions.json
├── .gitignore
└── README.md
```

---

## Pipeline

### 1. Data Preparation

The original financial dataset is reduced and processed to generate the datasets used by the different models.

Run:

```bash
python3 datos_orginales/PROCESING_DATOS.py
```

This generates:

```text
data/train_reduced.jsonl
data/validation_reduced.jsonl
data/test_reduced.jsonl
```

Data consistency can be checked with:

```bash
python3 datos_orginales/comprobaciones.py
```

---

## 2. Named Entity Recognition

The NER module identifies relevant entities in financial text.

To evaluate the saved model:

```bash
python3 ner/main.py
```

To train it from scratch:

```bash
cd ner
python3 -c "from main import train_main; train_main()"
cd ..
```

The trained model is stored in:

```text
ner/models/bilstm_ner.pt
```

---

## 3. Sentiment Analysis

Sentiment labels are generated using a financial sentiment model.

Run:

```bash
python3 sa/build_sentiment_finbert.py
```

The generated datasets are stored in:

```text
sa/reduced_data_with_sentiment/
```

To evaluate the saved sentiment model:

```bash
cd sa
python3 main_sa.py
cd ..
```

To train it:

```bash
cd sa
python3 -c "from main_sa import train_main; train_main()"
cd ..
```

The trained model is stored in:

```text
sa/models/bilstm_sa.pt
```

---

## 4. Joint NER + Sentiment Model

A multitask model combines both Named Entity Recognition and Sentiment Analysis.

The trained model is stored in:

```text
joint/models/bilstm_joint_ner_sa.pt
```

The main training implementation is located in:

```text
joint/train_joint.py
```

This model shares information between the two NLP tasks while producing separate NER and sentiment outputs.

---

## 5. Image Captioning

The multimodal component generates captions for images associated with selected financial examples.

First, extract the selected examples:

```bash
python3 extract_selected_alert_examples.py
```

Then generate image captions:

```bash
python3 image_captioning/generar_json.py
```

Generated captions are stored in:

```text
imagenes_generadas/captions.json
```

Finally, merge the financial examples with their captions:

```bash
python3 image_captioning/merge_captions.py
```

The combined dataset is stored in:

```text
selected_alert_train_examples_with_captions.json
```

---

## 6. Financial Alert Generation

The base financial alert pipeline is implemented in:

```text
alert/alert_main.py
```

It combines processed financial information with local language models to generate alerts.

Run:

```bash
python3 alert/alert_main.py
```

---

## 7. Multimodal Combination Analysis

The main comparison script is:

```bash
python3 alert/alert_main_combinaciones.py
```

The system supports seven different information combinations:

```text
1. NER only
2. Sentiment Analysis only
3. Image Captioning only
4. NER + Sentiment Analysis
5. NER + Sentiment Analysis + Image Captioning
6. NER + Image Captioning
7. Sentiment Analysis + Image Captioning
```

All combinations are evaluated using the same selected examples, allowing the effect of each information source to be compared.

Predictions are stored in files such as:

```text
data/alert_predictions_ner_only.json
data/alert_predictions_sa_only.json
data/alert_predictions_caption_only.json
data/alert_predictions_ner_sa.json
data/alert_predictions_ner_sa_caption.json
data/alert_predictions_ner_caption.json
data/alert_predictions_sa_caption.json
```

---

## Installation

Python 3.10+ is recommended.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install torch pandas scikit-learn transformers datasets tqdm ollama
```

---

## Ollama Setup

The image captioning and alert generation stages require a local Ollama server.

The project uses:

```text
llava-phi3
qwen2.5:3b
```

Download them with:

```bash
ollama pull llava-phi3
ollama pull qwen2.5:3b
```

Make sure Ollama is running before executing the multimodal alert generation pipeline.

---

## Quick Start

The repository includes trained model checkpoints for the main NLP components:

```text
ner/models/bilstm_ner.pt
sa/models/bilstm_sa.pt
joint/models/bilstm_joint_ner_sa.pt
```

If the required models and captions are already available, the final comparison can be executed directly with:

```bash
python3 alert/alert_main_combinaciones.py
```

Then select the desired information combination from the menu.

---

## Full Pipeline

To reproduce the complete workflow:

```text
1. Prepare the reduced financial dataset
2. Train or evaluate the NER model
3. Generate sentiment labels
4. Train or evaluate the sentiment model
5. Train the joint NER + sentiment model
6. Extract selected multimodal examples
7. Generate image captions
8. Merge captions with financial examples
9. Generate and compare financial alerts
```

---

## Main Outputs

The pipeline generates several intermediate and final outputs:

```text
data/train_reduced.jsonl
data/validation_reduced.jsonl
data/test_reduced.jsonl

ner/models/bilstm_ner.pt
sa/models/bilstm_sa.pt
joint/models/bilstm_joint_ner_sa.pt

selected_alert_train_examples.json
imagenes_generadas/captions.json
selected_alert_train_examples_with_captions.json

data/alert_predictions_*.json
```

---

## Project Highlights

This project demonstrates the integration of several areas of Artificial Intelligence within a single pipeline:

- Natural Language Processing
- Deep Learning
- Named Entity Recognition
- Sentiment Analysis
- Multitask Learning
- Multimodal AI
- Image Captioning
- Local Large Language Models
- Financial NLP
- AI-based information fusion

Rather than relying on a single model, the project explores how multiple sources of structured and unstructured information can be combined to produce richer financial outputs.

---

## Future Improvements

Potential extensions include:

- Transformer-based NER models
- Transformer-based multitask architectures
- Larger multimodal models
- Retrieval-Augmented Generation (RAG)
- Real-time financial news processing
- Automated alert delivery
- Cloud deployment
- Model performance dashboards
- Quantitative evaluation of generated alerts
- Integration with live financial data sources

---

## Academic Context

Developed as a collaborative final project involving **Natural Language Processing and Deep Learning** within the **Mathematical Engineering and Artificial Intelligence** program at **ICAI School of Engineering – Universidad Pontificia Comillas**.

The original development history and individual contributions are preserved in the Git commit history.

---

## Author Profile

**María Román Cantillana**

Mathematical Engineering & Artificial Intelligence  
ICAI School of Engineering – Universidad Pontificia Comillas

GitHub: [mariaromancanti](https://github.com/mariaromancanti)
