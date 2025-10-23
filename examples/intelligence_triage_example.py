"""
Intelligence Triage Example

This script demonstrates how to use the IntelligenceTriage classifier to classify
user queries into categories (L1 or L2).

Usage:
    python examples/intelligence_triage_example.py

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - Run from repository root directory
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flex_ml.models.intelligence_triage import IntelligenceTriage, triage_query


def example_l1_classification():
    """Example of L1 (6 categories) classification."""
    print("=" * 80)
    print("Intelligence Triage - L1 Classification Example")
    print("=" * 80)

    # Initialize the triage classifier for L1
    triage = IntelligenceTriage(classification_level="L1")

    # Example queries
    queries = [
        "입사 지원서는 어떻게 제출하나요?",
        "Device Trust 인증을 설정하는 방법을 알려주세요",
        "회의실 예약은 어떻게 하나요?",
        "1on1 미팅에서 팀원에게 어떤 질문을 해야 하나요?",
        "성과 평가는 언제 진행되나요?",
    ]

    print("\n질문 분류 결과 (L1):\n")
    for i, query in enumerate(queries, 1):
        category = triage.classify(query)
        print(f"{i}. 질문: {query}")
        print(f"   분류: {category}\n")


def example_l2_classification():
    """Example of L2 (13 categories) classification."""
    print("=" * 80)
    print("Intelligence Triage - L2 Classification Example")
    print("=" * 80)

    # Initialize the triage classifier for L2
    triage = IntelligenceTriage(classification_level="L2")

    # Example queries
    queries = [
        "입사 지원서는 어떻게 제출하나요?",
        "Device Trust 인증을 설정하는 방법을 알려주세요",
        "회의실 예약은 어떻게 하나요?",
        "1on1 미팅에서 팀원에게 어떤 질문을 해야 하나요?",
        "성과 평가는 언제 진행되나요?",
        "건강검진은 어떻게 신청하나요?",
        "회식 문화는 어떻게 되나요?",
    ]

    print("\n질문 분류 결과 (L2):\n")
    for i, query in enumerate(queries, 1):
        category = triage.classify(query)
        print(f"{i}. 질문: {query}")
        print(f"   분류: {category}\n")


def example_convenience_function():
    """Example using the convenience function."""
    print("=" * 80)
    print("Intelligence Triage - Convenience Function Example")
    print("=" * 80)

    query = "그라운드 피트니스 이용 방법을 알려주세요"

    print(f"\n질문: {query}\n")

    # L1 classification
    category_l1 = triage_query(query, classification_level="L1")
    print(f"L1 분류: {category_l1}")

    # L2 classification
    category_l2 = triage_query(query, classification_level="L2")
    print(f"L2 분류: {category_l2}")


def example_batch_classification():
    """Example of batch classification."""
    print("=" * 80)
    print("Intelligence Triage - Batch Classification Example")
    print("=" * 80)

    # Initialize the triage classifier
    triage = IntelligenceTriage(classification_level="L1")

    # Batch of queries
    queries = [
        "도서 구매는 어떻게 신청하나요?",
        "휴가를 사용하려면 어떻게 해야 하나요?",
        "리크루팅 식사 비용은 어떻게 청구하나요?",
        "비용 원칙은 무엇인가요?",
    ]

    print("\n배치 분류 결과:\n")
    categories = triage.classify_batch(queries)

    for query, category in zip(queries, categories):
        print(f"질문: {query}")
        print(f"분류: {category}\n")


def example_without_fewshot():
    """Example of classification without few-shot examples."""
    print("=" * 80)
    print("Intelligence Triage - Without Few-Shot Examples")
    print("=" * 80)

    triage = IntelligenceTriage(classification_level="L1")

    query = "주식매수선택권은 어떻게 행사하나요?"

    print(f"\n질문: {query}\n")

    # With few-shot examples (default)
    category_with = triage.classify(query, use_examples=True)
    print(f"Few-shot 사용 O: {category_with}")

    # Without few-shot examples
    category_without = triage.classify(query, use_examples=False)
    print(f"Few-shot 사용 X: {category_without}")


def main():
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Intelligence Triage Examples" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run examples
    try:
        example_l1_classification()
        print("\n")

        example_l2_classification()
        print("\n")

        example_convenience_function()
        print("\n")

        example_batch_classification()
        print("\n")

        example_without_fewshot()
        print("\n")

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
