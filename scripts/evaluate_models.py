"""
Model benchmark script.
Compares VADER+spaCy vs DeBERTa vs fine-tuned model on a common test set.

Usage: python scripts/evaluate_models.py
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ground truth test set — deterministic, labeled manually
TEST_SET: List[Tuple[str, str]] = [
    ("The battery lasts all day.", "positive"),
    ("Battery drains too fast.", "negative"),
    ("Battery is okay.", "neutral"),
    ("The camera takes amazing photos in daylight.", "positive"),
    ("Camera is blurry and disappointing.", "negative"),
    ("Screen is vibrant and sharp.", "positive"),
    ("Display looks washed out.", "negative"),
    ("Performance is buttery smooth.", "positive"),
    ("The phone lags on basic tasks.", "negative"),
    ("Great value for money.", "positive"),
    ("Way too overpriced.", "negative"),
    ("Build quality feels premium.", "positive"),
    ("Plastic body feels cheap.", "negative"),
    ("The software is intuitive.", "positive"),
    ("Too many bugs in the UI.", "negative"),
    ("Product arrived on time.", "positive"),
    ("Delivery took forever.", "negative"),
    ("Average performance overall.", "neutral"),
    ("Decent product, nothing special.", "neutral"),
    ("It's alright but could be better.", "neutral"),
]


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def benchmark_vader() -> dict:
    logger.info("Benchmarking VADER + spaCy...")
    from core.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer(use_bert=False)

    y_true, y_pred = [], []
    latencies = []

    for text, label in TEST_SET:
        t0 = time.perf_counter()
        result = analyzer.analyze_review(text)
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true.append(label)
        y_pred.append(result.get("overall_sentiment", "neutral"))

    metrics = _compute_metrics(y_true, y_pred)
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    metrics["model"] = "vader_spacy"
    return metrics


def benchmark_deberta() -> dict:
    logger.info("Benchmarking DeBERTa (optional — may be slow on CPU)...")
    try:
        from core.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer(use_bert=True)

        y_true, y_pred = [], []
        latencies = []
        for text, label in TEST_SET:
            t0 = time.perf_counter()
            result = analyzer.analyze_review(text)
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(label)
            y_pred.append(result.get("overall_sentiment", "neutral"))

        metrics = _compute_metrics(y_true, y_pred)
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        metrics["model"] = "deberta_absa"
        return metrics
    except Exception as e:
        logger.warning(f"DeBERTa benchmark failed: {e}")
        return {"model": "deberta_absa", "error": str(e)}


def benchmark_fine_tuned(model_path: str) -> dict:
    logger.info(f"Benchmarking fine-tuned model at: {model_path}")
    if not Path(model_path).exists():
        return {"model": "fine_tuned", "error": f"Model not found at {model_path}"}
    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline("text-classification", model=model_path, device=-1)

        y_true, y_pred = [], []
        latencies = []
        for text, label in TEST_SET:
            t0 = time.perf_counter()
            result = classifier(text)[0]
            latencies.append((time.perf_counter() - t0) * 1000)
            predicted = result["label"].lower()
            # Map to standard labels
            if "pos" in predicted:
                predicted = "positive"
            elif "neg" in predicted:
                predicted = "negative"
            else:
                predicted = "neutral"
            y_true.append(label)
            y_pred.append(predicted)

        metrics = _compute_metrics(y_true, y_pred)
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        metrics["model"] = "fine_tuned"
        return metrics
    except Exception as e:
        logger.error(f"Fine-tuned model benchmark failed: {e}")
        return {"model": "fine_tuned", "error": str(e)}


def run_benchmark(include_deberta: bool = False, fine_tuned_path: str = "models/fine_tuned/best_model"):
    results = []
    results.append(benchmark_vader())

    if include_deberta:
        results.append(benchmark_deberta())

    ft_result = benchmark_fine_tuned(fine_tuned_path)
    if "error" not in ft_result:
        results.append(ft_result)

    # Print summary table
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    for r in results:
        if "error" in r:
            print(f"\n[{r['model']}] ERROR: {r['error']}")
            continue
        print(f"\n[{r['model']}]")
        print(f"  Accuracy:   {r.get('accuracy', 'N/A')}")
        print(f"  Macro F1:   {r.get('macro_f1', 'N/A')}")
        print(f"  Avg Latency: {r.get('avg_latency_ms', 'N/A')} ms")
        if "report" in r:
            print(r["report"])

    # Save CSV
    output_path = Path("data/benchmark_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_results = [{k: v for k, v in r.items() if k != "report"} for r in results]
    if flat_results:
        fieldnames = list(flat_results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)
    logger.info(f"Benchmark results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_deberta", action="store_true")
    parser.add_argument("--fine_tuned_path", default="models/fine_tuned/best_model")
    args = parser.parse_args()
    run_benchmark(args.include_deberta, args.fine_tuned_path)
