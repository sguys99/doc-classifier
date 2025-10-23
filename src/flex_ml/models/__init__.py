"""
Models package for document classification and query generation.

This package contains:
- schemas: Pydantic models and enums for classification
- document_classifier: Main classifier implementation
- intelligence_triage: Query classifier for intelligence triage
- query_generator: Query generation for HR Q&A chatbot
"""

from flex_ml.models.document_classifier import DocumentClassifier, classify_document
from flex_ml.models.intelligence_triage import IntelligenceTriage, triage_query
from flex_ml.models.query_generator import QueryGenerator, generate_queries
from flex_ml.models.schemas import (
    CategoryL1,
    CategoryL2,
    DocumentCategoryL1,
    DocumentCategoryL2,
    QueryCategoryL1,
    QueryCategoryL2,
)

__all__ = [
    # Document Classifier
    "DocumentClassifier",
    "classify_document",
    # Intelligence Triage
    "IntelligenceTriage",
    "triage_query",
    # Query Generator
    "QueryGenerator",
    "generate_queries",
    # Schemas
    "CategoryL1",
    "CategoryL2",
    "DocumentCategoryL1",
    "DocumentCategoryL2",
    "QueryCategoryL1",
    "QueryCategoryL2",
]
