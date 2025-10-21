"""
Example usage of DocumentClassifier

This script demonstrates how to use the document classification function.
The classifier expects input in page_content format (markdown-style documents).
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flex_ml.document_classifier import DocumentClassifier, classify_document


def main():
    """Run classification examples."""

    # Example documents in page_content format (markdown-style)
    # These are realistic examples similar to the actual data format
    test_documents = [
        # Example 1: 구성원 여정 (채용 관련)
        """
## 리크루팅 식사 비용 지원

- 리크루팅 (채용 지원) 목적으로 외부 후보자(동료,지인 등)와 식사를 하실 경우 1인당 3만원까지의 비용을 지원하고 있습니다.

## 신청 방법
- 워크플로우를 통해 작성을 해주세요.
- 담당자 승인 후 비용이 지급됩니다.
        """,
        # Example 2: 지원 제도 (업무 지원)
        """
---
💡 사내 도서관 이용 및 도서 구입 지원 안내

### 도서 구입 신청 방법
1. 도서관 시스템에서 원하는 도서를 검색합니다.
2. 신청 버튼을 클릭하여 신청서를 작성합니다.
3. 승인 후 도서가 배송됩니다.

### 이용 시간
- 평일: 09:00 - 18:00
- 문의: #team-library
        """,
        # Example 3: 조직원칙 및 리더십
        """
# 리더십의 출발

플렉스팀의 리드는 동료를 인정하며 충분히 이해하고, 코칭하는 현명한 리더입니다.

💡 **[ 코칭의 3가지 철학 ]**

1. 모든 사람은 스스로 성장할 수 있는 가능성과 잠재력이 있습니다.
2. 동기부여는 강요가 아닌 자발적 참여에서 나옵니다.
3. 1 on 1 미팅을 통해 지속적으로 동료를 관찰하고 코칭합니다.
        """,
        # Example 4: 근무환경 및 제도
        """
## 근무/휴게시간 안내

### 근무 시간
- 오전 9시 ~ 오후 6시 (8시간 근무)
- 점심시간: 12시 ~ 1시 (1시간)

### 유연 근무제
- 코어타임: 10시 ~ 4시
- 자율 출퇴근 가능

### 재택근무
- 주 2회 재택근무 가능
- 사전 신청 필수
        """,
        # Example 5: 성장 및 발전
        """
## 온라인 교육 지원 프로그램

### 지원 내용
- 업무 관련 온라인 강의 수강료 100% 지원
- 연간 최대 50만원까지 지원

### 신청 방법
1. 수강하고자 하는 강의 정보 확인
2. 워크플로우에서 교육 지원 신청서 작성
3. 리더 승인 후 수강 시작
4. 수료 후 수료증 제출
        """,
    ]

    print("=" * 80)
    print("Document Classification Example")
    print("=" * 80)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print('export OPENAI_API_KEY="your-api-key-here"')
        return

    # Method 1: Using the convenience function
    print("\n## Method 1: Using classify_document() function\n")
    for i, doc in enumerate(test_documents[:3], 1):
        print(f"{i}. Document preview: {doc[:100].strip()}...")
        try:
            category = classify_document(doc, use_examples=True)
            print(f"   Category: {category}\n")
        except Exception as e:
            print(f"   Error: {e}\n")

    # Method 2: Using the DocumentClassifier class
    print("\n## Method 2: Using DocumentClassifier class\n")
    try:
        classifier = DocumentClassifier(model="gpt-4o-mini")

        for i, doc in enumerate(test_documents, 1):
            category = classifier.classify(doc, use_examples=True)
            print(f"{i}. Document preview: {doc[:80].strip()}...")
            print(f"   Category: {category}\n")

    except Exception as e:
        print(f"Error initializing classifier: {e}")

    # Method 3: Batch classification
    print("\n## Method 3: Batch classification\n")
    try:
        classifier = DocumentClassifier(model="gpt-4o-mini")
        categories = classifier.classify_batch(test_documents, use_examples=True)

        for i, (doc, category) in enumerate(zip(test_documents, categories), 1):
            print(f"{i}. Preview: {doc[:60].strip()}...")
            print(f"   Category: {category}\n")

    except Exception as e:
        print(f"Error in batch classification: {e}")


if __name__ == "__main__":
    main()