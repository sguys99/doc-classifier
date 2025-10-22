"""
Evaluate DocumentClassifier on the actual CSV data

This script evaluates the classifier's performance by:
1. Loading documents from the CSV file
2. Classifying each document using the page_content
3. Comparing predictions with actual category labels (L1 or L2)
4. Computing accuracy metrics

Supports both L1 (6 categories) and L2 (13 categories) classification.

Usage:
    # Evaluate L1 classification on all documents
    python evaluate_classifier.py --level L1

    # Evaluate L2 classification on 20 samples
    python evaluate_classifier.py --level L2 --sample-size 20

    # Evaluate without few-shot examples
    python evaluate_classifier.py --level L1 --no-examples
"""

from json import load
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flex_ml.document_classifier import DocumentClassifier
from flex_ml.utils.path import RAW_DATA_PATH


def evaluate_classifier(
    sample_size: int = None,
    use_examples: bool = True,
    random_seed: int = 42,
    classification_level: str = "L1",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate the classifier on CSV data.

    Args:
        sample_size: Number of samples to evaluate. If None, use all data.
        use_examples: Whether to use few-shot examples in classification.
        random_seed: Random seed for sampling.
        classification_level: "L1" for 6 categories or "L2" for 13 categories.

    Returns:
        Tuple of (results_df, metrics_dict)
    """
    # Load data
    data_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    df = pd.read_csv(data_path)

    print(f"Total documents in dataset: {len(df)}")
    print(f"Classification level: {classification_level}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)
        print(f"Evaluating on {sample_size} sampled documents")
    else:
        print(f"Evaluating on all {len(df)} documents")

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = DocumentClassifier(
        model="gpt-4o-mini", classification_level=classification_level
    )

    # Classify each document
    print("\nClassifying documents...")
    predictions = []
    actuals = []

    # Use the correct category column based on classification level
    category_column = "categoryL1" if classification_level == "L1" else "categoryL2"

    for idx, row in df.iterrows():
        page_content = row["page_content"]
        actual_category = row[category_column]

        try:
            predicted_category = classifier.classify(
                page_content, use_examples=use_examples
            )
            predictions.append(predicted_category)
            actuals.append(actual_category)

            print(
                f"[{len(predictions)}/{len(df)}] "
                f"Actual: {actual_category} | Predicted: {predicted_category} | "
                f"{'✓' if predicted_category == actual_category else '✗'}"
            )

        except Exception as e:
            print(f"[{len(predictions)+1}/{len(df)}] Error: {e}")
            predictions.append("ERROR")
            actuals.append(actual_category)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "page_id": df["page_id"].values,
            "title": df["title"].values,
            "actual": actuals,
            "predicted": predictions,
            "correct": [a == p for a, p in zip(actuals, predictions)],
        }
    )

    # Calculate metrics
    total = len(results_df)
    correct = results_df["correct"].sum()
    accuracy = correct / total if total > 0 else 0

    # Per-category accuracy
    category_metrics = {}
    for category in classifier.CATEGORIES:
        category_df = results_df[results_df["actual"] == category]
        if len(category_df) > 0:
            category_correct = category_df["correct"].sum()
            category_accuracy = category_correct / len(category_df)
            category_metrics[category] = {
                "total": len(category_df),
                "correct": category_correct,
                "accuracy": category_accuracy,
            }

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "category_metrics": category_metrics,
    }

    return results_df, metrics


def print_evaluation_report(results_df: pd.DataFrame, metrics: Dict):
    """Print evaluation metrics in a readable format."""

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall accuracy
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")

    # Per-category accuracy
    print("\n" + "-" * 80)
    print("Per-Category Performance:")
    print("-" * 80)

    for category, cat_metrics in metrics["category_metrics"].items():
        print(
            f"{category:25s} | "
            f"Accuracy: {cat_metrics['accuracy']:.2%} "
            f"({cat_metrics['correct']}/{cat_metrics['total']})"
        )

    # Show misclassified examples
    print("\n" + "-" * 80)
    print("Misclassified Documents:")
    print("-" * 80)

    misclassified = results_df[~results_df["correct"]]
    if len(misclassified) == 0:
        print("No misclassifications! Perfect accuracy!")
    else:
        for idx, row in misclassified.iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"  Actual:    {row['actual']}")
            print(f"  Predicted: {row['predicted']}")

    # Confusion matrix
    print("\n" + "-" * 80)
    print("Confusion Matrix:")
    print("-" * 80)

    from collections import defaultdict

    confusion = defaultdict(lambda: defaultdict(int))
    for _, row in results_df.iterrows():
        confusion[row["actual"]][row["predicted"]] += 1

    # Print header
    categories = sorted(set(results_df["actual"].unique()))
    print(f"{'Actual \\ Predicted':25s} | " + " | ".join([f"{c[:10]:10s}" for c in categories]))
    print("-" * 80)

    for actual in categories:
        row_str = f"{actual:25s} | "
        for predicted in categories:
            count = confusion[actual][predicted]
            row_str += f"{count:10d} | "
        print(row_str)


def main():
    """Run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Document Classifier")
    parser.add_argument(
        "--level",
        type=str,
        default="L1",
        choices=["L1", "L2"],
        help="Classification level: L1 (6 categories) or L2 (13 categories)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all documents)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Disable few-shot examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Document Classifier Evaluation")
    print("=" * 80)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print('export OPENAI_API_KEY="your-api-key-here"')
        return

    # Run evaluation
    try:
        results_df, metrics = evaluate_classifier(
            sample_size=args.sample_size,
            use_examples=not args.no_examples,
            random_seed=args.seed,
            classification_level=args.level,
        )

        # Print report
        print_evaluation_report(results_df, metrics)

        # Save results to CSV
        filename = f"evaluation_results_{args.level.lower()}.csv"
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "intermediate", filename
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()