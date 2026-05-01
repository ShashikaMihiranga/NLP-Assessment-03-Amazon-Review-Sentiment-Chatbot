"""
Text augmentation pipeline for minority-class oversampling.

Pipeline per sample:
    back_translate() → bert_augment() → quality_filter()

Usage:
    from src.augmentation_pipeline import augment_dataframe

    df_aug = augment_dataframe(
        df,
        text_col="text",
        label_col="label",
        output_path="data/processed/augmented.csv",
    )
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Back-translation ──────────────────────────────────────────────────────────

def back_translate(
    text: str,
    delay_range: tuple[float, float] = (0.5, 1.5),
) -> str:
    """
    Translate English → German → English using deep-translator.

    Returns the original text unchanged if any translation call fails.
    A random delay is inserted between the two API calls to avoid rate limiting.
    """
    try:
        from deep_translator import GoogleTranslator

        german = GoogleTranslator(source="en", target="de").translate(text)
        if not german:
            return text
        time.sleep(random.uniform(*delay_range))

        english = GoogleTranslator(source="de", target="en").translate(german)
        time.sleep(random.uniform(*delay_range))

        return english if english else text

    except Exception as exc:
        logger.warning("back_translate failed (%s) — returning original.", exc)
        return text


# ── BERT contextual augmentation ──────────────────────────────────────────────

def _build_bert_augmenter(aug_p: float = 0.175):
    """
    Build a ContextualWordEmbsAug instance with bert-base-uncased.

    aug_p is kept in the 0.15–0.20 range so only 15–20% of tokens are replaced.
    """
    import torch
    import nlpaug.augmenter.word as naw

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading BERT augmenter on %s …", device)
    return naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased",
        action="substitute",
        aug_p=aug_p,
        device=device,
    )


def bert_augment(text: str, augmenter) -> str:
    """
    Apply BERT contextual word substitution to *text*.

    Returns the original text if augmentation raises an exception.
    """
    try:
        result = augmenter.augment(text)
        return result[0] if isinstance(result, list) else result
    except Exception as exc:
        logger.warning("bert_augment failed (%s) — returning input.", exc)
        return text


# ── Semantic-similarity quality filter ───────────────────────────────────────

def _build_sim_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer (all-MiniLM-L6-v2) …")
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_similarity(original: str, augmented: str, sim_model) -> float:
    from sentence_transformers import util

    emb_orig = sim_model.encode(original, convert_to_tensor=True)
    emb_aug  = sim_model.encode(augmented, convert_to_tensor=True)
    return float(util.cos_sim(emb_orig, emb_aug))


def passes_quality_filter(
    original: str,
    augmented: str,
    sim_model,
    threshold: float = 0.75,
) -> tuple[bool, float]:
    """
    Return (passed, similarity_score).

    Keeps the augmented sample only when cosine similarity >= threshold.
    """
    score = _cosine_similarity(original, augmented, sim_model)
    return score >= threshold, score


# ── Single-sample pipeline ────────────────────────────────────────────────────

def augment_one(
    text: str,
    augmenter,
    sim_model,
    sim_threshold: float,
    delay_range: tuple[float, float],
) -> Optional[str]:
    """
    Run the full pipeline on a single text sample.

    back_translate → bert_augment → quality_filter

    Returns the augmented string, or None if the quality filter rejects it.
    """
    bt_text  = back_translate(text, delay_range)
    aug_text = bert_augment(bt_text, augmenter)
    passed, score = passes_quality_filter(text, aug_text, sim_model, sim_threshold)

    if not passed:
        logger.debug(
            "Sample discarded (similarity=%.3f < threshold=%.2f): %.60s …",
            score, sim_threshold, text,
        )
        return None

    return aug_text


# ── Main pipeline ─────────────────────────────────────────────────────────────

def augment_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    imbalance_ratio: float = 0.5,
    augmentations_per_sample: int = 2,
    sim_threshold: float = 0.75,
    aug_p: float = 0.175,
    delay_range: tuple[float, float] = (0.5, 1.5),
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment minority-class samples and return the combined dataframe.

    Parameters
    ----------
    df                       Input dataframe.
    text_col                 Column containing the raw text.
    label_col                Column containing the class label.
    imbalance_ratio          Classes whose (count / majority_count) < this value
                             are treated as minority classes.
    augmentations_per_sample Exact number of accepted augmented copies to produce
                             per original sample. A retry budget of 4× is used.
    sim_threshold            Minimum cosine similarity (0–1) to keep a sample.
    aug_p                    Fraction of words replaced by BERT (keep in 0.15–0.20).
    delay_range              (min_s, max_s) random sleep between translation calls.
    output_path              If given, the result is saved to this CSV path.
    random_state             Seed for reproducibility.

    Returns
    -------
    DataFrame with original rows (is_augmented=False) plus accepted synthetic
    rows (is_augmented=True).
    """
    random.seed(random_state)
    np.random.seed(random_state)

    # ── Identify minority classes ─────────────────────────────────────────────
    label_counts    = df[label_col].value_counts()
    max_count       = label_counts.max()
    minority_labels = label_counts[
        label_counts / max_count < imbalance_ratio
    ].index.tolist()

    logger.info("Original class counts:\n%s", label_counts.to_string())

    if not minority_labels:
        logger.info(
            "No minority classes found (ratio threshold=%.2f). "
            "Returning original dataframe.",
            imbalance_ratio,
        )
        result = df.copy()
        result["is_augmented"] = False
        return result

    logger.info(
        "Minority classes (count/max < %.2f): %s",
        imbalance_ratio, minority_labels,
    )

    # ── Lazy-load heavy models once ───────────────────────────────────────────
    augmenter = _build_bert_augmenter(aug_p)
    sim_model = _build_sim_model()

    df_out = df.copy()
    df_out["is_augmented"] = False

    discarded_counts: dict = {lbl: 0 for lbl in minority_labels}
    new_rows: list[dict]   = []

    # ── Augment each minority-class sample ────────────────────────────────────
    for label in minority_labels:
        subset = df[df[label_col] == label]
        logger.info(
            "Class '%s': %d original samples × %d augmentations each …",
            label, len(subset), augmentations_per_sample,
        )

        for original_idx, row in subset.iterrows():
            original_text = str(row[text_col])
            generated     = 0
            attempts      = 0
            max_attempts  = augmentations_per_sample * 4

            while generated < augmentations_per_sample and attempts < max_attempts:
                attempts += 1
                aug_text = augment_one(
                    original_text, augmenter, sim_model, sim_threshold, delay_range
                )

                if aug_text is None:
                    discarded_counts[label] += 1
                    continue

                # Reject trivial no-ops (text unchanged after full pipeline)
                if aug_text.strip().lower() == original_text.strip().lower():
                    discarded_counts[label] += 1
                    continue

                new_row = row.to_dict()
                new_row[text_col]       = aug_text
                new_row["is_augmented"] = True
                new_rows.append(new_row)
                generated += 1

            if generated < augmentations_per_sample:
                logger.warning(
                    "Class '%s', row %s: accepted %d/%d augmentations "
                    "(budget exhausted after %d attempts).",
                    label, original_idx, generated, augmentations_per_sample, attempts,
                )

    # ── Combine and save ──────────────────────────────────────────────────────
    if new_rows:
        df_out = pd.concat(
            [df_out, pd.DataFrame(new_rows)], ignore_index=True
        )

    _print_summary(df, df_out, label_col, minority_labels, discarded_counts)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)
        logger.info("Saved %d rows to %s", len(df_out), out)

    return df_out


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(
    df_orig: pd.DataFrame,
    df_out: pd.DataFrame,
    label_col: str,
    minority_labels: list,
    discarded_counts: dict,
) -> None:
    orig_counts = df_orig[label_col].value_counts().sort_index()
    aug_counts  = df_out[label_col].value_counts().sort_index()

    all_labels = aug_counts.index.tolist()
    col_w      = max(len(str(l)) for l in all_labels) + 3

    print("\n" + "=" * 65)
    print("  AUGMENTATION SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<{col_w}} {'Original':>10}  {'Augmented':>10}  {'Added':>7}")
    print("  " + "-" * 61)

    for lbl in all_labels:
        orig  = orig_counts.get(lbl, 0)
        aug   = aug_counts[lbl]
        delta = aug - orig
        tag   = " *" if lbl in minority_labels else "  "
        print(f"  {str(lbl) + tag:<{col_w}} {orig:>10,}  {aug:>10,}  {delta:>+7,}")

    print("  " + "-" * 61)
    total_orig = len(df_orig)
    total_aug  = len(df_out)
    print(f"  {'TOTAL':<{col_w}} {total_orig:>10,}  {total_aug:>10,}  "
          f"{total_aug - total_orig:>+7,}")

    print("\n  Discarded by quality filter (* = minority class):")
    for lbl, count in discarded_counts.items():
        print(f"    {lbl}: {count} sample(s) rejected (similarity < threshold)")

    print("=" * 65 + "\n")


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "text": [
                "This product works great, very happy with the purchase.",
                "Excellent quality, fast delivery, would buy again.",
                "Good value for money, solid build quality.",
                "Terrible product, broke after one day.",          # minority: negative
                "Completely disappointed, waste of money.",        # minority: negative
                "Average product, nothing special to say.",        # minority: neutral
            ],
            "label": [
                "positive", "positive", "positive",
                "negative", "negative",
                "neutral",
            ],
        }
    )

    df_result = augment_dataframe(
        sample_df,
        text_col="text",
        label_col="label",
        imbalance_ratio=0.7,
        augmentations_per_sample=2,
        sim_threshold=0.75,
        aug_p=0.175,
        delay_range=(0.3, 0.8),
        output_path="data/processed/augmented_sample.csv",
        random_state=42,
    )

    print(df_result[["label", "is_augmented", "text"]].to_string(index=False))
