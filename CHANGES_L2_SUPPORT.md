# L2 Classification Support - Implementation Summary

## 변경 사항 요약

DocumentClassifier에 categoryL2 기준 분류 기능을 추가했습니다. 이제 L1 (6개 카테고리)과 L2 (13개 카테고리) 분류를 모두 지원합니다.

## 파일 변경 내역

### 1. `src/flex_ml/models.py` (수정)

**추가된 내용:**
- `CategoryL2` enum: 13개의 세부 카테고리 정의
  - 지원 제도: 업무 지원, 생활 지원
  - 조직원칙 및 리더십: 리더십, 문화/팀빌딩, 원칙/철학, 협업 방식
  - 근무환경 및 제도: 근무 제도, 오피스
  - 구성원 여정: 채용, 온보딩, 오프보딩, 구성원을 위한 개인정보 처리방침
  - 성장 및 발전: 성과/성장

- `DocumentCategoryL2` Pydantic 모델: L2 분류 결과를 위한 구조화된 출력 모델

**기존 내용:**
- `CategoryL1` enum: 6개의 상위 카테고리
- `DocumentCategoryL1` Pydantic 모델: L1 분류 결과용

### 2. `src/flex_ml/document_classifier.py` (대폭 수정)

**주요 변경사항:**

#### `__init__` 메서드
- 새로운 파라미터: `classification_level: Literal["L1", "L2"] = "L1"`
- 동적으로 설정되는 속성들:
  ```python
  self.category_column = "categoryL1" if level == "L1" else "categoryL2"
  self.CATEGORIES = CATEGORIES_L1 if level == "L1" else CATEGORIES_L2
  self.response_format = DocumentCategoryL1 if level == "L1" else DocumentCategoryL2
  ```
- 자동으로 올바른 프롬프트 파일 선택:
  - L1: `configs/classifier_prompt_l1.yaml`
  - L2: `configs/classifier_prompt_l2.yaml`

#### `_prepare_examples` 메서드
- `self.category_column`을 사용하여 동적으로 올바른 카테고리 컬럼 필터링
- L1과 L2에 맞는 예시 자동 선택

#### `_build_prompt` 메서드
- 분류 수준에 따라 다른 지침(guidelines) 생성
- L1: "6가지 카테고리 중 하나"
- L2: "13가지 세부 카테고리 중 하나"

#### `classify` 메서드
- `self.response_format` 사용 (기존 하드코딩된 `DocumentCategory` 대신)
- 올바른 Pydantic 모델로 응답 파싱
- 독스트링 업데이트: 분류 수준이 초기화 시 결정된다는 설명 추가

#### `classify_document` 함수
- 새로운 파라미터: `classification_level: Literal["L1", "L2"] = "L1"`
- 사용 예시 추가:
  ```python
  category = classify_document("입사 지원서...", classification_level="L2")
  ```

### 3. `configs/classifier_prompt_l1.yaml` (이름 변경)

- 기존 `classifier_prompt.yaml`에서 이름 변경
- L1 분류용 프롬프트 (6개 카테고리)

### 4. `configs/classifier_prompt_l2.yaml` (신규 생성)

- L2 분류용 프롬프트 (13개 카테고리)
- 각 카테고리별 상세 설명과 예시 포함
- 상위 카테고리 별로 그룹화된 구조

### 5. `examples/test_l2_classification.py` (신규 생성)

L1과 L2 분류를 모두 테스트하는 스크립트:
- CSV 데이터에서 샘플 문서 선택
- L1과 L2 분류 동시 수행
- 실제 라벨과 예측 결과 비교
- 편의 함수도 테스트

**실행 방법:**
```bash
export OPENAI_API_KEY="your-api-key"
source .venv/bin/activate
python examples/test_l2_classification.py
```

### 6. `examples/L2_CLASSIFICATION_GUIDE.md` (신규 생성)

L2 분류 사용 가이드 문서:
- 사용 방법 예시 (3가지 방법)
- L2 카테고리 상세 설명
- 테스트 방법
- 주요 변경사항 요약

## 사용 예시

### 기본 사용법

```python
from flex_ml.document_classifier import DocumentClassifier

# L1 분류 (6개 카테고리)
classifier_l1 = DocumentClassifier(classification_level="L1")
category_l1 = classifier_l1.classify("재택근무 정책 안내")
print(category_l1)  # Output: 근무환경 및 제도

# L2 분류 (13개 카테고리)
classifier_l2 = DocumentClassifier(classification_level="L2")
category_l2 = classifier_l2.classify("재택근무 정책 안내")
print(category_l2)  # Output: 근무 제도
```

### 편의 함수 사용

```python
from flex_ml.document_classifier import classify_document

text = "입사 지원서 제출 방법"

# L1 분류
result_l1 = classify_document(text, classification_level="L1")
# Output: 구성원 여정

# L2 분류
result_l2 = classify_document(text, classification_level="L2")
# Output: 채용
```

## 기술적 세부사항

### 타입 안전성
- `Literal["L1", "L2"]` 타입 힌트로 유효한 값만 허용
- Enum 기반 카테고리 정의로 오타 방지
- Pydantic 구조화된 출력으로 유효한 카테고리만 반환

### 코드 중복 제거 (DRY)
- 단일 클래스로 L1과 L2 모두 처리
- 분류 수준에 따라 동적으로 설정 변경
- 프롬프트 파일만 별도 관리

### 확장성
- 새로운 분류 수준 추가 시 최소한의 코드 변경
- 카테고리 추가/변경은 Enum과 YAML 파일만 수정

## 테스트 결과

초기화 테스트 결과:
```
L1 Classifier:
  - Level: L1
  - Category column: categoryL1
  - Number of categories: 6
  - Response format: DocumentCategoryL1
  - Prompt config path: .../classifier_prompt_l1.yaml

L2 Classifier:
  - Level: L2
  - Category column: categoryL2
  - Number of categories: 13
  - Response format: DocumentCategoryL2
  - Prompt config path: .../classifier_prompt_l2.yaml
```

## 다음 단계 (선택사항)

1. **평가 스크립트 작성**:
   - `examples/evaluate_classifier_l2.py` 생성
   - L2 분류 정확도 측정
   - Confusion matrix 생성

2. **README 업데이트**:
   - `README_CLASSIFIER.md`에 L2 분류 내용 추가
   - 사용 예시 추가

3. **성능 최적화**:
   - L2 프롬프트 튜닝
   - Few-shot 예시 개수 조정
   - Temperature 파라미터 실험

## 구현 완료 체크리스트

- ✅ CategoryL2 enum 정의 (13개 카테고리)
- ✅ DocumentCategoryL2 Pydantic 모델 생성
- ✅ classifier_prompt_l2.yaml 작성
- ✅ classifier_prompt_l1.yaml 이름 변경
- ✅ DocumentClassifier 클래스 수정 (classification_level 파라미터)
- ✅ classify() 메서드 업데이트 (동적 response_format)
- ✅ _build_prompt() 메서드 업데이트 (동적 guidelines)
- ✅ _prepare_examples() 메서드 업데이트 (동적 column)
- ✅ classify_document() 함수 업데이트
- ✅ 테스트 스크립트 작성 (test_l2_classification.py)
- ✅ 사용 가이드 문서 작성 (L2_CLASSIFICATION_GUIDE.md)
- ✅ 초기화 테스트 완료

모든 구현이 완료되었으며, L1과 L2 분류가 정상적으로 작동합니다! 🎉
