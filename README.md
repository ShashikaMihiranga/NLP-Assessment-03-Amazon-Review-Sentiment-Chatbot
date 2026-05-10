# Amazon Electronics Reviews — Aspect-Based Sentiment Analysis & RAG Chatbot

An end-to-end NLP pipeline built on 6.7 million Amazon Electronics reviews. The project
extracts product aspects and their sentiment, aggregates them into a product scoring table,
and exposes the results through a natural-language RAG chatbot powered by a local LLM.

---

## Project Pipeline

```
Raw Reviews (6.7M)
       │
       ▼
   EDA & Feature Engineering
   (ratings, text quality, reviewer behaviour)
       │
       ▼
   N-gram Analysis (NLTK) + Tokenization & POS Tagging (spaCy)
       │
       ▼
   Classical Baselines — TF-IDF + SMOTE
   ├── Complement Naive Bayes
   └── Logistic Regression
       │
       ▼
   Text Augmentation
   (back-translation EN→DE→EN + BERT fill-mask + similarity filter)
       │
       ▼
   DeepSeek-R1:8b (local, zero-shot)
   └── Labels 1,500 reviews with aspect-sentiment pairs
       │
       ▼
   DistilBERT Fine-tuning (ABSA)
   └── Trained on DeepSeek labels → scales to 300K reviews
       │
       ▼
   Product Scoring Table (9,636 products × 12 aspects)
       │
       ▼
   RAG Chatbot (DeepSeek-R1:8b via Ollama)
   └── Answers natural-language product queries
```

---

## Models & Techniques

| Model / Technique | Role |
| --- | --- |
| NLTK bigrams & trigrams | Language pattern analysis per sentiment class |
| spaCy `en_core_web_sm` | Tokenization, POS tagging, named entity recognition |
| TF-IDF + Complement Naive Bayes | Baseline sentiment classifier |
| TF-IDF + Logistic Regression + SMOTE | Stronger baseline with class balancing |
| BERT fill-mask (`bert-base-uncased`) | Contextual word substitution for augmentation |
| `all-MiniLM-L6-v2` | Semantic similarity filter for augmentation quality |
| DeepSeek-R1:8b (Ollama) | Zero-shot ABSA labelling + RAG chatbot |
| DistilBERT (fine-tuned) | Aspect-based sentiment at 300K scale |

---

## Project Structure

```
NPL Assement 03/
├── notebooks/
│   └── main.ipynb                # Full pipeline (Steps 1–25, 79 cells)
├── src/
│   └── augmentation_pipeline.py  # Back-translation + BERT augmentation module
├── models/
│   └── distilbert_absa/
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│       # model.safetensors excluded (255 MB) — retrain via Step 21
├── reports/
│   ├── class_distribution_before_smote.png
│   ├── confusion_matrix.png
│   ├── distilbert_training_curve.png
│   ├── naive_bayes_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   └── model_comparison.png
├── data/
│   ├── processed/                # Generated files (gitignored — too large)
│   └── sample/
│       └── sample_reviews.csv
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install and start Ollama (for DeepSeek steps)

```bash
# Install from https://ollama.com
ollama pull deepseek-r1:8b
ollama serve
```

### 3. Run the notebook

Open `notebooks/main.ipynb` and run cells in order.

| Step | Description | Output |
| --- | --- | --- |
| 1–12 | Load raw data, EDA, merge reviews + metadata | `electronics_nlp_ready.csv` |
| 13b | NLTK n-grams + spaCy tokenization | Charts in `reports/` |
| 13–17 | Feature engineering, vocabulary analysis, class imbalance | — |
| 17b | Naive Bayes + Logistic Regression baselines | Confusion matrices, model comparison chart |
| 18 | Text augmentation *(auto-skips if already done)* | `electronics_augmented.csv` |
| 19 | DeepSeek aspect extraction on 1,500 reviews | `aspect_results.csv` |
| 20 | Build 300K stratified inference sample | `sample_300k.csv` |
| 21 | Fine-tune DistilBERT *(auto-skips if model exists)* | `models/distilbert_absa/` |
| 22 | Batch ABSA inference on 300K reviews | `absa_inference_300k.csv` |
| 23 | Aggregate product scoring table + heatmap | `product_scores.csv` |
| 24 | RAG chatbot demo queries | Console output |
| 25 | Model discussion & future improvements | — |

> **Steps 18 and 21 auto-skip** if their output files already exist on disk.
> Delete the relevant file to force a re-run.

---

## Key Results

- **9,636 products** scored across **12 aspects** (battery life, build quality, performance, etc.)
- DistilBERT fine-tuned on ~3,000 DeepSeek-labelled aspect-sentiment pairs
- Chatbot example queries:
  - *"Which product has the best battery life score?"*
  - *"Which products have the worst build quality?"*
  - *"What are the top 3 products for value for money?"*
  - *"Which brand has the highest overall score?"*

---

## Data

Raw data files are not included in this repository (too large for GitHub).

| File | Size | Source |
| --- | --- | --- |
| `Electronics_5.json.gz` | ~1.5 GB | Amazon Reviews dataset |
| `meta_Electronics.jsonl.gz` | ~3.5 GB | Amazon Product Metadata |

Processed files (`electronics_nlp_ready.csv`, `sample_300k.csv`, etc.) are generated
by running the notebook from Step 1.

---

## Requirements

Key dependencies (see `requirements.txt` for the full list):

- Python 3.11+
- `torch >= 2.0`, `transformers >= 4.31`
- `scikit-learn`, `imbalanced-learn`
- `nltk`, `spacy`
- `sentence-transformers`, `deep-translator`
- Ollama running locally with `deepseek-r1:8b` pulled
