"""
Data models for classification.

This module defines the Pydantic models and Enums used for document or query classification.
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


class CategoryL2(str, Enum):
    """Valid categoryL2 values for document classification."""

    # 지원 제도
    WORK_SUPPORT = "업무 지원"
    LIFE_SUPPORT = "생활 지원"

    # 조직원칙 및 리더십
    LEADERSHIP = "리더십"
    CULTURE_TEAM_BUILDING = "문화/ 팀빌딩"
    PRINCIPLES_PHILOSOPHY = "원칙/ 철학"
    COLLABORATION = "협업 방식"

    # 근무환경 및 제도
    WORK_POLICY = "근무 제도"
    OFFICE = "오피스"

    # 구성원 여정
    RECRUITMENT = "채용"
    ONBOARDING = "온보딩"
    OFFBOARDING = "오프보딩"
    PRIVACY_POLICY = "구성원을 위한 개인정보 처리방침"

    # 성장 및 발전
    PERFORMANCE_GROWTH = "성과/성장"

    # 기타
    OTHER = "기타"


class DocumentCategoryL1(BaseModel):
    """Structured output for L1 document classification."""

    category: CategoryL1 = Field(
        ...,
        description=(
            "The classified L1 category for the document. " "Must be one of the predefined categories."
        ),
    )


class DocumentCategoryL2(BaseModel):
    """Structured output for L2 document classification."""

    category: CategoryL2 = Field(
        ...,
        description=(
            "The classified L2 category for the document. " "Must be one of the predefined categories."
        ),
    )


class QueryCategoryL1(BaseModel):
    """Structured output for L1 query classification (Intelligence Triage)."""

    category: CategoryL1 = Field(
        ...,
        description="The classified L1 category for the query. Must be one of the predefined categories.",
    )


class QueryCategoryL2(BaseModel):
    """Structured output for L2 query classification (Intelligence Triage)."""

    category: CategoryL2 = Field(
        ...,
        description="The classified L2 category for the query. Must be one of the predefined categories.",
    )
