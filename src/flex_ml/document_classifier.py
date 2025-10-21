"""
Document Classification Module using LLM Models

This module provides functions to classify text documents into predefined categories
using LLM models with few-shot prompting and Pydantic structured outputs.
"""

import os
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from flex_ml.utils.path import RAW_DATA_PATH


class CategoryL1(str, Enum):
    """Valid categoryL1 values."""

    SUPPORT_SYSTEM = "지원 제도"
    ORGANIZATION_LEADERSHIP = "조직원칙 및 리더십"
    WORK_ENVIRONMENT = "근무환경 및 제도"
    MEMBER_JOURNEY = "구성원 여정"
    GROWTH_DEVELOPMENT = "성장 및 발전"


class DocumentCategory(BaseModel):
    """Structured output for document classification."""

    category: CategoryL1 = Field(
        ...,
        description="The classified category for the document. Must be one of the predefined categories.",
    )


class DocumentClassifier:
    """
    Classifier for categorizing documents into categoryL1 using LLM.

    Categories:
    - 지원 제도: Support systems and benefits
    - 조직원칙 및 리더십: Organizational principles and leadership
    - 근무환경 및 제도: Work environment and policies
    - 구성원 여정: Member journey (recruitment, onboarding, etc.)
    - 성장 및 발전: Growth and development
    """

    CATEGORIES = [
        "지원 제도",
        "조직원칙 및 리더십",
        "근무환경 및 제도",
        "구성원 여정",
        "성장 및 발전",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        data_path: Optional[str] = None,
    ):
        """
        Initialize the DocumentClassifier.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
            model: OpenAI model to use. Default is "gpt-4o-mini".
            data_path: Path to CSV data file. If None, uses default from RAW_DATA_PATH.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # Load training data for few-shot examples
        self.data_path = data_path or os.path.join(
            RAW_DATA_PATH, "people_intelligence_documents.csv"
        )
        self.df = pd.read_csv(self.data_path)
        self.category_examples = self._prepare_examples()

    def _prepare_examples(self, examples_per_category: int = 3) -> Dict[str, List[Dict]]:
        """
        Prepare few-shot examples from the dataset.

        Args:
            examples_per_category: Number of examples per category to use.

        Returns:
            Dictionary mapping category names to example documents.
        """
        category_examples = {}

        for category in self.CATEGORIES:
            examples = (
                self.df[self.df["categoryL1"] == category]
                .head(examples_per_category)[["title", "page_content"]]
                .to_dict("records")
            )
            category_examples[category] = examples

        return category_examples

    def _build_prompt(self, text: str, use_examples: bool = True) -> str:
        """
        Build the classification prompt for OpenAI API.

        Args:
            text: Input text to classify (page_content format).
            use_examples: Whether to include few-shot examples in the prompt.

        Returns:
            Formatted prompt string.
        """
        prompt = """당신은 회사 내부 문서를 분류하는 전문가입니다.
        주어진 문서 내용(page_content)을 분석하여 다음 5가지 카테고리 중 하나로 분류해주세요:
        
        1. 지원 제도: 업무 지원, 생활 지원, 복지 혜택 등에 관한 내용
        2. 조직원칙 및 리더십: 리더십, 원칙/철학, 문화/팀빌딩에 관한 내용
        3. 근무환경 및 제도: 근무 제도, 오피스 환경, 근무 시간 등에 관한 내용
        4. 구성원 여정: 채용, 온보딩, 퇴사 등 구성원 라이프사이클에 관한 내용
        5. 성장 및 발전: 교육, 학습, 커리어 개발 등에 관한 내용
        """

        if use_examples:
            prompt += "\n## 각 카테고리별 예시 문서:\n\n"
            for category, examples in self.category_examples.items():
                prompt += f"### {category}\n"
                for i, example in enumerate(examples[:2], 1):  # Use 2 examples per category
                    content = example.get("page_content", "")[:500]  # Show more content
                    prompt += f"예시 {i}:\n{content}...\n\n"

        prompt += f"""
        ## 분류할 문서 내용:
        # {text}
         
         
        ## 지침:
          - 위 문서 내용의 주제와 맥락을 분석하여 가장 적합한 카테고리를 선택하세요.
          - 반드시 위 5가지 카테고리 중 하나만 정확히 응답하세요.
          """

        return prompt


    def classify(
        self,
        text: str,
        use_examples: bool = True,
        temperature: float = 0.0,
    ) -> str:
        """
        Classify a text document into one of the categoryL1 categories using Pydantic structured output.

        Args:
            text: Input text to classify (page_content format).
            use_examples: Whether to use few-shot examples. Default is True.
            temperature: OpenAI temperature parameter. Default is 0.0 for deterministic output.

        Returns:
            Classified category name (one of the CATEGORIES).

        Raises:
            ValueError: If the classification result is not in the valid categories.
        """
        prompt = self._build_prompt(text, use_examples=use_examples)

        # Use Pydantic structured output to ensure valid category
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=DocumentCategory,
            temperature=temperature,
        )

        # Extract the category from the structured response
        result = completion.choices[0].message.parsed
        return result.category.value

    def classify_batch(
        self, texts: List[str], use_examples: bool = True, **kwargs
    ) -> List[str]:
        """
        Classify multiple text documents.

        Args:
            texts: List of input texts to classify.
            use_examples: Whether to use few-shot examples. Default is True.
            **kwargs: Additional arguments passed to classify().

        Returns:
            List of classified category names.
        """
        return [self.classify(text, use_examples=use_examples, **kwargs) for text in texts]


def classify_document(
    text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_examples: bool = True,
) -> str:
    """
    Convenience function to classify a single document.

    Args:
        text: Input text to classify.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
        model: OpenAI model to use. Default is "gpt-4o-mini".
        use_examples: Whether to use few-shot examples. Default is True.

    Returns:
        Classified category name.

    Example:
        >>> category = classify_document("입사 지원서 제출 방법에 대한 안내")
        >>> print(category)
        구성원 여정
    """
    classifier = DocumentClassifier(api_key=api_key, model=model)
    return classifier.classify(text, use_examples=use_examples)