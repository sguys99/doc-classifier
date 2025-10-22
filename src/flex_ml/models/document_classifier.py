"""
Document Classification Module using LLM Models

This module provides functions to classify text documents into predefined categories
using LLM models with few-shot prompting and Pydantic structured outputs.

Supports both L1 (6 categories) and L2 (13 categories) classification.
"""

import os
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml
from openai import OpenAI

from flex_ml.models.schemas import (
    CategoryL1,
    CategoryL2,
    DocumentCategoryL1,
    DocumentCategoryL2,
)
from flex_ml.utils.path import CONFIG_PATH, RAW_DATA_PATH


class DocumentClassifier:
    """
    Classifier for categorizing documents into categoryL1 or categoryL2 using LLM.

    Supports two classification levels:
    - L1 (6 categories): 지원 제도, 조직원칙 및 리더십, 근무환경 및 제도, 구성원 여정, 성장 및 발전, 기타
    - L2 (13 categories): 업무 지원, 생활 지원, 리더십, 문화/ 팀빌딩, etc.
    """

    CATEGORIES_L1 = [category.value for category in CategoryL1]
    CATEGORIES_L2 = [category.value for category in CategoryL2]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        data_path: Optional[str] = None,
        classification_level: Literal["L1", "L2"] = "L1",
        prompt_config_path: Optional[str] = None,
    ):
        """
        Initialize the DocumentClassifier.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
            model: OpenAI model to use. Default is "gpt-4o-mini".
            data_path: Path to CSV data file. If None, uses default from RAW_DATA_PATH.
            classification_level: "L1" for 6 categories or "L2" for 13 categories. Default is "L1".
            prompt_config_path: Path to prompt config YAML file. If None, uses default based on level.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.classification_level = classification_level

        # Set category column name based on level
        self.category_column = "categoryL1" if classification_level == "L1" else "categoryL2"

        # Set CATEGORIES based on level
        self.CATEGORIES = self.CATEGORIES_L1 if classification_level == "L1" else self.CATEGORIES_L2

        # Set response format based on level
        self.response_format = (
            DocumentCategoryL1 if classification_level == "L1" else DocumentCategoryL2
        )

        # Load prompt configuration
        if prompt_config_path is None:
            prompt_filename = (
                "classifier_prompt_l1.yaml"
                if classification_level == "L1"
                else "classifier_prompt_l2.yaml"
            )
            self.prompt_config_path = os.path.join(CONFIG_PATH, prompt_filename)
        else:
            self.prompt_config_path = prompt_config_path

        self.prompt_config = self._load_prompt_config()

        # Load training data for few-shot examples
        self.data_path = data_path or os.path.join(
            RAW_DATA_PATH, "people_intelligence_documents.csv"
        )
        self.df = pd.read_csv(self.data_path)
        self.category_examples = self._prepare_examples()

    def _load_prompt_config(self) -> Dict:
        """
        Load prompt configuration from YAML file.

        Returns:
            Dictionary containing prompt configuration.
        """
        with open(self.prompt_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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
                self.df[self.df[self.category_column] == category]
                .head(examples_per_category)[["title", "page_content"]]
                .to_dict("records")
            )
            category_examples[category] = examples

        return category_examples

    def _build_prompt(self, text: str, use_examples: bool = True) -> str:
        """
        Build the classification prompt for OpenAI API using YAML configuration.

        Args:
            text: Input text to classify (page_content format).
            use_examples: Whether to include few-shot examples in the prompt.

        Returns:
            Formatted prompt string.
        """
        # Load base prompt from YAML
        prompt = self.prompt_config["base_prompt"]

        # Add few-shot examples from CSV
        if use_examples:
            prompt += "\n\n## 각 카테고리별 예시 문서:\n\n"
            for category, examples in self.category_examples.items():
                if examples:  # Only if examples exist
                    prompt += f"### {category}\n"
                    for i, example in enumerate(examples[:2], 1):  # Use 2 examples per category
                        content = example.get("page_content", "")[:500]  # Show more content
                        prompt += f"예시 {i}:\n{content}...\n\n"

        # Add document to classify
        num_categories = len(self.CATEGORIES)
        if self.classification_level == "L1":
            guidelines = f"""## 지침:
  - 위 문서 내용의 주제와 맥락을 분석하여 가장 적합한 카테고리를 선택하세요.
  - 반드시 위 {num_categories}가지 카테고리 중 하나만 정확히 응답하세요.
  - 1-5번 카테고리에 명확히 속한다면 해당 카테고리를 선택하고, 어디에도 속하지 않는다면 6. 기타를 선택하세요.
"""
        else:  # L2
            guidelines = f"""## 지침:
  - 위 문서 내용의 주제와 맥락을 분석하여 가장 적합한 카테고리를 선택하세요.
  - 반드시 위 {num_categories}가지 세부 카테고리 중 하나만 정확히 응답하세요.
  - 문서의 구체적인 내용을 바탕으로 가장 관련성이 높은 카테고리를 선택하세요.
"""

        prompt += f"""
## 분류할 문서 내용:
{text}

{guidelines}"""

        return prompt

    def classify(
        self,
        text: str,
        use_examples: bool = True,
        temperature: float = 0.0,
    ) -> str:
        """
        Classify a text document into one of the categories using Pydantic structured output.

        The classification level (L1 or L2) is determined by the classification_level parameter
        passed during initialization.

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
            response_format=self.response_format,
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
    classification_level: Literal["L1", "L2"] = "L1",
) -> str:
    """
    Convenience function to classify a single document.

    Args:
        text: Input text to classify.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
        model: OpenAI model to use. Default is "gpt-4o-mini".
        use_examples: Whether to use few-shot examples. Default is True.
        classification_level: "L1" for 6 categories or "L2" for 13 categories. Default is "L1".

    Returns:
        Classified category name.

    Example:
        >>> category = classify_document("입사 지원서 제출 방법에 대한 안내")
        >>> print(category)
        구성원 여정

        >>> category = classify_document("입사 지원서 제출 방법에 대한 안내", classification_level="L2")
        >>> print(category)
        채용
    """
    classifier = DocumentClassifier(api_key=api_key, model=model, classification_level=classification_level)
    return classifier.classify(text, use_examples=use_examples)
