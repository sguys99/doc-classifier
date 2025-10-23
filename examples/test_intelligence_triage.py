"""
Simple test script for Intelligence Triage

This script tests the basic functionality of the IntelligenceTriage classifier.

Usage:
    python examples/test_intelligence_triage.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_import():
    """Test that imports work correctly."""
    print("Testing imports...")
    try:
        from flex_ml.models import IntelligenceTriage, triage_query
        from flex_ml.models.schemas import CategoryL1, CategoryL2, QueryCategoryL1, QueryCategoryL2

        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_l1_classification():
    """Test L1 classification."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⊘ Skipping L1 classification test (OPENAI_API_KEY not set)")
        return True

    print("\nTesting L1 classification...")
    try:
        from flex_ml.models import triage_query

        query = "Device Trust 인증 방법을 알려주세요"
        category = triage_query(query, classification_level="L1")

        assert category in [
            "지원 제도",
            "조직원칙 및 리더십",
            "근무환경 및 제도",
            "구성원 여정",
            "성장 및 발전",
            "기타",
        ], f"Invalid L1 category: {category}"

        print(f"✓ L1 classification successful: {query} -> {category}")
        return True
    except Exception as e:
        print(f"✗ L1 classification failed: {e}")
        return False


def test_l2_classification():
    """Test L2 classification."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⊘ Skipping L2 classification test (OPENAI_API_KEY not set)")
        return True

    print("\nTesting L2 classification...")
    try:
        from flex_ml.models import triage_query

        query = "건강검진은 어떻게 신청하나요?"
        category = triage_query(query, classification_level="L2")

        assert category in [
            "업무 지원",
            "생활 지원",
            "리더십",
            "문화/ 팀빌딩",
            "원칙/ 철학",
            "협업 방식",
            "근무 제도",
            "오피스",
            "채용",
            "온보딩",
            "오프보딩",
            "구성원을 위한 개인정보 처리방침",
            "성과/성장",
        ], f"Invalid L2 category: {category}"

        print(f"✓ L2 classification successful: {query} -> {category}")
        return True
    except Exception as e:
        print(f"✗ L2 classification failed: {e}")
        return False


def test_batch_classification():
    """Test batch classification."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⊘ Skipping batch classification test (OPENAI_API_KEY not set)")
        return True

    print("\nTesting batch classification...")
    try:
        from flex_ml.models import IntelligenceTriage

        triage = IntelligenceTriage(classification_level="L1")
        queries = ["도서는 어떻게 신청하나요?", "휴가 신청 방법을 알려주세요"]
        categories = triage.classify_batch(queries)

        assert len(categories) == len(queries), "Number of categories doesn't match queries"
        print(f"✓ Batch classification successful: processed {len(queries)} queries")
        return True
    except Exception as e:
        print(f"✗ Batch classification failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Intelligence Triage Test Suite")
    print("=" * 60)

    tests = [
        test_import,
        test_l1_classification,
        test_l2_classification,
        test_batch_classification,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
