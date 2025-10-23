"""
Intelligence Triage Module using LLM Models

This module provides functions to classify user queries into predefined categories
using LLM models with few-shot prompting and Pydantic structured outputs.

Supports both L1 (6 categories) and L2 (14 categories) classification for queries.
"""

import os
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml
from openai import OpenAI

from flex_ml.models.schemas import (
    CategoryL1,
    CategoryL2,
    QueryCategoryL1,
    QueryCategoryL2,
)
from flex_ml.utils.path import CONFIG_PATH, RAW_DATA_PATH


class IntelligenceTriage:
    """
    Classifier for categorizing user queries into categoryL1 or categoryL2 using LLM.

    This class is designed for intelligence triage - routing user queries to appropriate
    categories based on their intent and content.

    Supports two classification levels:
    - L1 (6 categories): 지원 제도, 조직원칙 및 리더십, 근무환경 및 제도, 구성원 여정, 성장 및 발전, 기타
    - L2 (14 categories): 업무 지원, 생활 지원, 리더십, 문화/ 팀빌딩, etc.
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
        Initialize the IntelligenceTriage classifier.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
            model: OpenAI model to use. Default is "gpt-4o-mini".
            data_path: Path to CSV data file with queries. If None, uses default path.
            classification_level: "L1" for 6 categories or "L2" for 13 categories.
            prompt_config_path: Path to prompt config YAML file. If None, uses default.
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
        self.response_format = QueryCategoryL1 if classification_level == "L1" else QueryCategoryL2

        # Load prompt configuration
        if prompt_config_path is None:
            prompt_filename = (
                "triage_prompt_l1.yaml" if classification_level == "L1" else "triage_prompt_l2.yaml"
            )
            self.prompt_config_path = os.path.join(CONFIG_PATH, prompt_filename)
        else:
            self.prompt_config_path = prompt_config_path

        self.prompt_config = self._load_prompt_config()

        # Load training data for few-shot examples
        self.data_path = data_path or os.path.join(
            RAW_DATA_PATH, "people_intelligence_documents_with_queries.csv"
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
            Dictionary mapping category names to example queries.
        """
        category_examples = {}

        for category in self.CATEGORIES:
            examples = (
                self.df[self.df[self.category_column] == category]
                .head(examples_per_category)[["query", self.category_column]]
                .to_dict("records")
            )
            category_examples[category] = examples

        return category_examples

    def _build_prompt(self, query: str, use_examples: bool = True) -> str:
        """
        Build the classification prompt for OpenAI API using YAML configuration.

        Args:
            query: User query to classify.
            use_examples: Whether to include few-shot examples in the prompt.

        Returns:
            Formatted prompt string.
        """
        # Load base prompt from YAML
        prompt = self.prompt_config["base_prompt"]

        # Add few-shot examples from CSV
        if use_examples:
            prompt += "\n\n## 각 카테고리별 예시 질문:\n\n"
            for category, examples in self.category_examples.items():
                if examples:  # Only if examples exist
                    prompt += f"### {category}\n"
                    for i, example in enumerate(examples, 1):
                        example_query = example.get("query", "")
                        prompt += f"예시 {i}: {example_query}\n"
                    prompt += "\n"

        # Add query to classify
        num_categories = len(self.CATEGORIES)
        if self.classification_level == "L1":
            guidelines = f"""## 지침:
  - 위 질문의 의도와 내용을 분석하여 가장 적합한 카테고리를 선택하세요.
  - 반드시 위 {num_categories}가지 카테고리 중 하나만 정확히 응답하세요.
  - 1-5번 카테고리에 명확히 속한다면 해당 카테고리를 선택하고, 어디에도 속하지 않는다면 6. 기타를 선택하세요.
"""
        else:  # L2
            guidelines = f"""## 지침:
  - 위 질문의 의도와 내용을 분석하여 가장 적합한 카테고리를 선택하세요.
  - 반드시 위 {num_categories}가지 세부 카테고리 중 하나만 정확히 응답하세요.
  - 질문의 구체적인 내용을 바탕으로 가장 관련성이 높은 카테고리를 선택하세요.
"""

        prompt += f"""
## 분류할 질문:
{query}

{guidelines}"""

        return prompt

    def classify(
        self,
        query: str,
        use_examples: bool = True,
        temperature: float = 0.0,
    ) -> str:
        """
        Classify a user query into one of the categories using Pydantic structured output.

        The classification level (L1 or L2) is determined by the classification_level
        parameter passed during initialization.

        Args:
            query: User query to classify.
            use_examples: Whether to use few-shot examples. Default is True.
            temperature: OpenAI temperature parameter. Default is 0.0 for deterministic.

        Returns:
            Classified category name (one of the CATEGORIES).

        Raises:
            ValueError: If the classification result is not in the valid categories.
        """
        prompt = self._build_prompt(query, use_examples=use_examples)

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

    def classify_batch(self, queries: List[str], use_examples: bool = True, **kwargs) -> List[str]:
        """
        Classify multiple queries.

        Args:
            queries: List of user queries to classify.
            use_examples: Whether to use few-shot examples. Default is True.
            **kwargs: Additional arguments passed to classify().

        Returns:
            List of classified category names.
        """
        return [self.classify(query, use_examples=use_examples, **kwargs) for query in queries]


def triage_query(
    query: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_examples: bool = True,
    classification_level: Literal["L1", "L2"] = "L1",
) -> str:
    """
    Convenience function to classify a single query (Intelligence Triage).

    Args:
        query: User query to classify.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
        model: OpenAI model to use. Default is "gpt-4o-mini".
        use_examples: Whether to use few-shot examples. Default is True.
        classification_level: "L1" for 6 categories or "L2" for 13 categories.

    Returns:
        Classified category name.

    Example:
        >>> category = triage_query("입사 지원서는 어떻게 제출하나요?")
        >>> print(category)
        구성원 여정

        >>> category = triage_query("입사 지원서는 어떻게 제출하나요?", classification_level="L2")
        >>> print(category)
        채용
    """
    triage = IntelligenceTriage(api_key=api_key, model=model, classification_level=classification_level)
    return triage.classify(query, use_examples=use_examples)
