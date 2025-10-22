"""
Models package for document classification.

This package contains:
- schemas: Pydantic models and enums for classification
- document_classifier: Main classifier implementation
"""

from flex_ml.models.document_classifier import DocumentClassifier, classify_document
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
    # Schemas
    "CategoryL1",
    "CategoryL2",
    "DocumentCategoryL1",
    "DocumentCategoryL2",
]
