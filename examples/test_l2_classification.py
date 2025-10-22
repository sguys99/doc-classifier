"""
Test script to verify L2 classification support.

This script demonstrates and tests both L1 and L2 classification.
"""

import os
import sys

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flex_ml.models import DocumentClassifier
from flex_ml.utils.path import RAW_DATA_PATH


def test_classification_levels():
    """Test both L1 and L2 classification on sample documents."""
    # Load data
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv"))

    # Test documents - select ones with different L2 categories
    test_cases = [
        {"index": 0, "title": df.iloc[0]["title"]},
        {"index": 10, "title": df.iloc[10]["title"]},
        {"index": 20, "title": df.iloc[20]["title"]},
    ]

    print("=" * 80)
    print("Testing L1 and L2 Classification")
    print("=" * 80)

    for case in test_cases:
        idx = case["index"]
        row = df.iloc[idx]

        print(f"\n\n문서 제목: {row['title']}")
        print(f"문서 내용 (처음 200자): {row['page_content'][:200]}...")
        print("-" * 80)

        # Test L1 classification
        print("\n[L1 Classification - 6 categories]")
        classifier_l1 = DocumentClassifier(classification_level="L1")
        predicted_l1 = classifier_l1.classify(row["page_content"])
        actual_l1 = row["categoryL1"]

        print(f"  실제 카테고리 (L1): {actual_l1}")
        print(f"  예측 카테고리 (L1): {predicted_l1}")
        print(f"  결과: {'✓ 정답' if predicted_l1 == actual_l1 else '✗ 오답'}")

        # Test L2 classification
        print("\n[L2 Classification - 13 categories]")
        classifier_l2 = DocumentClassifier(classification_level="L2")
        predicted_l2 = classifier_l2.classify(row["page_content"])
        actual_l2 = row["categoryL2"]

        print(f"  실제 카테고리 (L2): {actual_l2}")
        print(f"  예측 카테고리 (L2): {predicted_l2}")
        print(f"  결과: {'✓ 정답' if predicted_l2 == actual_l2 else '✗ 오답'}")

        print("=" * 80)


def test_convenience_function():
    """Test the convenience function with both levels."""
    from flex_ml.models import classify_document

    test_text = "입사 지원서 제출 방법 및 면접 일정 안내"

    print("\n\n" + "=" * 80)
    print("Testing Convenience Function")
    print("=" * 80)
    print(f"\n테스트 텍스트: {test_text}\n")

    # L1 classification
    category_l1 = classify_document(test_text, classification_level="L1")
    print(f"L1 분류 결과: {category_l1}")

    # L2 classification
    category_l2 = classify_document(test_text, classification_level="L2")
    print(f"L2 분류 결과: {category_l2}")

    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print("\n🚀 Starting L1 and L2 Classification Tests\n")

    # Test classification levels on real data
    test_classification_levels()

    # Test convenience function
    test_convenience_function()

    print("\n\n✅ All tests completed!\n")
