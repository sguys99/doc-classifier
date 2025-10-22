"""
Models package for document classification and query generation.

This package contains:
- schemas: Pydantic models and enums for classification
- document_classifier: Main classifier implementation
- query_generator: Query generation for HR Q&A chatbot
"""

from flex_ml.models.document_classifier import DocumentClassifier, classify_document
from flex_ml.models.query_generator import QueryGenerator, generate_queries
from flex_ml.models.schemas import (
    CategoryL1,
    CategoryL2,
    DocumentCategoryL1,
    DocumentCategoryL2,
)

__all__ = [
    # Classifier
    "DocumentClassifier",
    "classify_document",
    # Query Generator
    "QueryGenerator",
    "generate_queries",
    # Schemas
    "CategoryL1",
    "CategoryL2",
    "DocumentCategoryL1",
    "DocumentCategoryL2",
]
