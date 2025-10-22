"""
Simple script to test classifier on a single document from CSV

This script demonstrates how to:
1. Load a single document from the CSV
2. Classify it using the page_content
3. Compare with the actual categoryL1
"""

import os
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flex_ml.models import DocumentClassifier
from flex_ml.utils.path import RAW_DATA_PATH


def test_single_document(index: int = 0):
    """
    Test classifier on a single document from the CSV.

    Args:
        index: Row index of the document to test (default: 0)
    """
    # Load data
    data_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    df = pd.read_csv(data_path)

    print(f"Total documents in dataset: {len(df)}")
    print(f"Testing document at index: {index}\n")

    # Get the document
    row = df.iloc[index]

    print("=" * 80)
    print("DOCUMENT INFO")
    print("=" * 80)
    print(f"Title: {row['title']}")
    print(f"Actual Category: {row['categoryL1']}")
    print(f"Category L2: {row['categoryL2']}")
    print(f"\nPage Content Preview (first 500 characters):")
    print("-" * 80)
    print(row["page_content"][:500])
    print("...")
    print("=" * 80)

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = DocumentClassifier(model="gpt-4o-mini")

    # Classify
    print("\nClassifying document...")
    try:
        predicted_category = classifier.classify(
            row["page_content"], use_examples=True
        )

        print("\n" + "=" * 80)
        print("CLASSIFICATION RESULT")
        print("=" * 80)
        print(f"Actual Category:    {row['categoryL1']}")
        print(f"Predicted Category: {predicted_category}")
        print(f"\nResult: {'✓ CORRECT' if predicted_category == row['categoryL1'] else '✗ INCORRECT'}")
        print("=" * 80)

        return predicted_category == row["categoryL1"]

    except Exception as e:
        print(f"\nError during classification: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_samples(num_samples: int = 5, random_seed: int = 42):
    """
    Test classifier on multiple random samples.

    Args:
        num_samples: Number of samples to test
        random_seed: Random seed for sampling
    """
    # Load data
    data_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    df = pd.read_csv(data_path)

    print(f"Testing {num_samples} random samples from {len(df)} documents\n")

    # Sample documents
    samples = df.sample(n=num_samples, random_state=random_seed)

    # Initialize classifier
    print("Initializing classifier...")
    classifier = DocumentClassifier(model="gpt-4o-mini")

    # Test each sample
    results = []
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'=' * 80}")
        print(f"Sample {i}/{num_samples}")
        print(f"{'=' * 80}")
        print(f"Title: {row['title']}")
        print(f"Actual: {row['categoryL1']}")

        try:
            predicted = classifier.classify(row["page_content"], use_examples=True)
            correct = predicted == row["categoryL1"]

            print(f"Predicted: {predicted}")
            print(f"Result: {'✓ CORRECT' if correct else '✗ INCORRECT'}")

            results.append(
                {
                    "title": row["title"],
                    "actual": row["categoryL1"],
                    "predicted": predicted,
                    "correct": correct,
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            results.append(
                {
                    "title": row["title"],
                    "actual": row["categoryL1"],
                    "predicted": "ERROR",
                    "correct": False,
                }
            )

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0

    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result["correct"] else "✗"
        print(f"{i}. {status} {result['title'][:50]:50s} | {result['actual']:20s} → {result['predicted']}")


def main():
    """Run tests."""

    print("=" * 80)
    print("Document Classifier Test")
    print("=" * 80)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print('export OPENAI_API_KEY="your-api-key-here"')
        return

    # Choose test mode
    print("\nTest Modes:")
    print("1. Test single document (index 0)")
    print("2. Test 5 random samples")
    print("3. Test specific document by index")

    try:
        choice = input("\nEnter choice (1-3) or press Enter for default [2]: ").strip()

        if choice == "1":
            test_single_document(index=0)
        elif choice == "3":
            index = int(input("Enter document index: "))
            test_single_document(index=index)
        else:  # Default to option 2
            test_multiple_samples(num_samples=5, random_seed=42)

    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()