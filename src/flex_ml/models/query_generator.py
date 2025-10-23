"""
Query Generation Module for HR Helpdesk service

This module provides functions to generate expected questions from HR document content for evaluation.
"""

import os
from typing import List, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


class GeneratedQueries(BaseModel):
    """Pydantic model for structured query generation output."""

    queries: List[str]


class QueryGenerator:
    """
    Generator for creating expected HR questions from document content.

    Uses LLM to analyze page_content and generate relevant questions
    that users might ask about that content.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        num_queries: int = 3,
    ):
        """
        Initialize the QueryGenerator.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
            model: OpenAI model to use. Default is "gpt-4o-mini".
            num_queries: Number of queries to generate per document. Default is 3.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.num_queries = num_queries

    def _build_prompt(self, page_content: str, title: str = "") -> str:
        """
        Build the query generation prompt for OpenAI API.

        Args:
            page_content: Document content to analyze.
            title: Document title (optional).

        Returns:
            Formatted prompt string.
        """
        prompt = f"""당신은 HR 질의응답 챗봇을 위한 학습 데이터를 생성하는 전문가입니다.

아래 HR 문서 내용을 읽고, 직원들이 이 정보에 대해 물어볼 법한 자연스러운 질문들을 생성해주세요.

## 문서 제목:
{title if title else "제목 없음"}

## 문서 내용:
{page_content}

## 지침:
1. 문서 내용을 기반으로 직원들이 실제로 물어볼 법한 {self.num_queries}개의 질문을 생성하세요
2. 질문은 자연스럽고 구체적이어야 합니다
3. 질문은 한국어로 작성하세요
4. 다양한 관점과 수준의 질문을 포함하세요 (예: 절차 문의, 자격 요건, 기한, 방법 등)
5. 문서에서 답을 찾을 수 있는 질문이어야 합니다

{self.num_queries}개의 질문을 생성해주세요."""

        return prompt

    def generate(
        self,
        page_content: str,
        title: str = "",
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Generate queries from a document's page_content.

        Args:
            page_content: Document content to analyze.
            title: Document title (optional).
            temperature: OpenAI temperature parameter. Default is 0.7 for creative output.

        Returns:
            List of generated queries.
        """
        prompt = self._build_prompt(page_content, title)

        # Use Pydantic structured output to ensure valid format
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=GeneratedQueries,
            temperature=temperature,
        )

        # Extract the queries from the structured response
        result = completion.choices[0].message.parsed
        return result.queries

    def generate_batch(
        self,
        documents: List[dict],
        temperature: float = 0.7,
    ) -> List[List[str]]:
        """
        Generate queries for multiple documents.

        Args:
            documents: List of dictionaries containing 'page_content' and optionally 'title'.
            temperature: OpenAI temperature parameter. Default is 0.7.

        Returns:
            List of query lists (one list per document).
        """
        results = []
        for doc in documents:
            page_content = doc.get("page_content", "")
            title = doc.get("title", "")
            queries = self.generate(page_content, title, temperature)
            results.append(queries)
        return results


def generate_queries(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    num_queries: int = 3,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Generate queries for a DataFrame containing HR documents.

    Reads page_content column and generates expected questions,
    adding them to a new 'query' column.

    Args:
        df: DataFrame with 'page_content' column (and optionally 'title' column).
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env variable.
        model: OpenAI model to use. Default is "gpt-4o-mini".
        num_queries: Number of queries to generate per document. Default is 3.
        temperature: OpenAI temperature parameter. Default is 0.7.

    Returns:
        DataFrame with added 'query' column containing list of generated queries.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("people_intelligence_documents.csv")
        >>> df_with_queries = generate_queries(df)
        >>> print(df_with_queries[['title', 'query']].head())
    """
    if "page_content" not in df.columns:
        raise ValueError("DataFrame must contain 'page_content' column")

    generator = QueryGenerator(api_key=api_key, model=model, num_queries=num_queries)

    # Prepare documents
    documents = df[["page_content"]].copy()
    if "title" in df.columns:
        documents["title"] = df["title"]
    else:
        documents["title"] = ""

    # Generate queries
    queries = []
    for idx, row in documents.iterrows():
        print(f"Generating queries for document {idx + 1}/{len(documents)}...")
        query_list = generator.generate(
            page_content=row["page_content"],
            title=row["title"],
            temperature=temperature,
        )
        queries.append(query_list)

    # Add query column to original DataFrame
    result_df = df.copy()
    result_df["query"] = queries

    return result_df
