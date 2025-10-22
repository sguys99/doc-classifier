"""
Data models for document classification.

This module defines the Pydantic models and Enums used for document classification.
"""

from enum import Enum

from pydantic import BaseModel, Field


class CategoryL1(str, Enum):
    """Valid categoryL1 values for document classification."""

    SUPPORT_SYSTEM = "지원 제도"
    ORGANIZATION_LEADERSHIP = "조직원칙 및 리더십"
    WORK_ENVIRONMENT = "근무환경 및 제도"
    MEMBER_JOURNEY = "구성원 여정"
    GROWTH_DEVELOPMENT = "성장 및 발전"
    OTHER = "기타"


class DocumentCategory(BaseModel):
    """Structured output for document classification."""

    category: CategoryL1 = Field(
        ...,
        description="The classified category for the document. Must be one of the predefined categories.",
    )
