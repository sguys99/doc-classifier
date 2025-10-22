# L2 Classification Guide

이 문서는 categoryL2 기준 분류를 위한 가이드입니다.

## 개요

DocumentClassifier는 이제 두 가지 분류 수준을 지원합니다:

- **L1 (6 categories)**: 지원 제도, 조직원칙 및 리더십, 근무환경 및 제도, 구성원 여정, 성장 및 발전, 기타
- **L2 (13 categories)**: 업무 지원, 생활 지원, 리더십, 문화/팀빌딩, 원칙/철학, 협업 방식, 근무 제도, 오피스, 채용, 온보딩, 오프보딩, 구성원을 위한 개인정보 처리방침, 성과/성장

## 사용 방법

### 방법 1: DocumentClassifier 클래스 사용

```python
from flex_ml.models import DocumentClassifier

# L1 분류 (6개 카테고리)
classifier_l1 = DocumentClassifier(classification_level="L1")
category_l1 = classifier_l1.classify("재택근무 정책 안내")
print(f"L1 분류: {category_l1}")  # Output: 근무환경 및 제도

# L2 분류 (13개 카테고리)
classifier_l2 = DocumentClassifier(classification_level="L2")
category_l2 = classifier_l2.classify("재택근무 정책 안내")
print(f"L2 분류: {category_l2}")  # Output: 근무 제도
```

### 방법 2: classify_document 편의 함수 사용

```python
from flex_ml.models import classify_document

text = "입사 지원서 제출 방법 및 면접 일정 안내"

# L1 분류
category_l1 = classify_document(text, classification_level="L1")
print(f"L1: {category_l1}")  # Output: 구성원 여정

# L2 분류
category_l2 = classify_document(text, classification_level="L2")
print(f"L2: {category_l2}")  # Output: 채용
```

### 방법 3: 배치 분류

```python
classifier_l2 = DocumentClassifier(classification_level="L2")

texts = [
    "재택근무 정책 및 출퇴근 시간 규정",
    "1 on 1 미팅 가이드 및 피드백 방법",
    "온라인 강의 수강 지원 프로그램",
]

categories = classifier_l2.classify_batch(texts)
for text, category in zip(texts, categories):
    print(f"{text} → {category}")
```

## 카테고리 상세 설명

### L2 카테고리 구조

L2 카테고리는 L1 카테고리의 세부 분류입니다:

#### 지원 제도 → 2개
- 업무 지원: IT 기기, 사원증, 도서, 출장, 퀵서비스, 주차 등
- 생활 지원: 피트니스, 건강검진, 의료비, 자녀 지원, 보험 등

#### 조직원칙 및 리더십 → 4개
- 리더십: 1on1 미팅, 동기부여, 코칭 방법론
- 문화/ 팀빌딩: 회식 문화, Health Check, Story Session
- 원칙/ 철학: 비용 원칙, 의사결정 원칙
- 협업 방식: 팀 간 협업, 커뮤니케이션 방법

#### 근무환경 및 제도 → 2개
- 근무 제도: 근무/휴게시간, 휴가, 유연근무, 재택근무
- 오피스: 회의실, 냉장고, 공간 이용, 좌석 배치

#### 구성원 여정 → 4개
- 채용: 리크루팅, 사내추천, 채용 프로세스
- 온보딩: 신규 입사자 환영, 오리엔테이션
- 오프보딩: 퇴사 절차
- 구성원을 위한 개인정보 처리방침: 구성원 관련 개인정보 보호

#### 성장 및 발전 → 1개
- 성과/성장: 성과 평가, Job Level, 1on1 Ground Rule, 총 보상, 주식매수선택권

## 테스트 방법

### 기본 테스트
```bash
# L2 분류 테스트 실행
python examples/test_l2_classification.py
```

### 평가 스크립트 (예정)
```bash
# L2 분류 정확도 평가 (구현 예정)
python examples/evaluate_classifier_l2.py
```

## 프롬프트 구성 파일

- `configs/classifier_prompt_l1.yaml`: L1 분류용 프롬프트
- `configs/classifier_prompt_l2.yaml`: L2 분류용 프롬프트

프롬프트를 수정하려면 해당 YAML 파일을 편집하세요.

## 주요 변경사항

1. **models.py**:
   - `CategoryL2` enum 추가 (13개 카테고리)
   - `DocumentCategoryL2` Pydantic 모델 추가

2. **document_classifier.py**:
   - `classification_level` 파라미터 추가 (`"L1"` 또는 `"L2"`)
   - 분류 수준에 따라 동적으로 카테고리, 응답 형식, 프롬프트 선택
   - `classify()` 메서드가 `self.response_format`을 사용하여 올바른 Pydantic 모델 적용

3. **prompt 파일**:
   - `classifier_prompt_l1.yaml`: 기존 프롬프트 (6 카테고리)
   - `classifier_prompt_l2.yaml`: 새로운 L2 프롬프트 (13 카테고리)

## 성능 고려사항

- L2 분류는 더 세밀한 구분을 요구하므로 L1보다 정확도가 다소 낮을 수 있습니다
- Few-shot examples를 사용하면 정확도가 향상됩니다
- `temperature=0.0`으로 설정하여 일관성 있는 결과를 얻을 수 있습니다
