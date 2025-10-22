# Document Classifier

OpenAI API와 프롬프트 엔지니어링을 활용한 문서 분류 시스템입니다.

## 개요

이 분류기는 텍스트를 입력받아 다음 5가지 카테고리 중 하나로 분류합니다:

1. **지원 제도**: 업무 지원, 생활 지원, 복지 혜택 등
2. **조직원칙 및 리더십**: 리더십, 원칙/철학, 문화/팀빌딩
3. **근무환경 및 제도**: 근무 제도, 오피스 환경, 근무 시간
4. **구성원 여정**: 채용, 온보딩, 퇴사 등 구성원 라이프사이클
5. **성장 및 발전**: 교육, 학습, 커리어 개발

## 기능

- **Few-shot Learning**: CSV 데이터의 실제 예시를 활용한 프롬프트 구성
- **단일/배치 분류**: 단일 문서 또는 여러 문서 동시 분류 지원
- **유연한 설정**: OpenAI 모델 선택, temperature 조정 등
- **간편한 API**: 함수 호출 또는 클래스 인스턴스 두 가지 방식 지원

## 설치 및 설정

### 1. 환경 설정

```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 의존성 설치

프로젝트의 가상환경에 이미 필요한 패키지가 설치되어 있습니다:
- `openai`
- `pandas`

## 사용 방법

### 방법 1: 간편한 함수 호출

```python
from flex_ml.document_classifier import classify_document

# 단일 문서 분류
text = "입사 지원서 제출 방법 및 면접 일정 안내"
category = classify_document(text)
print(category)  # Output: 구성원 여정
```

### 방법 2: 클래스 인스턴스 활용

```python
from flex_ml.document_classifier import DocumentClassifier

# 분류기 초기화
classifier = DocumentClassifier(
    api_key="your-api-key",  # Optional, 환경변수에서 자동 로드
    model="gpt-4o-mini",     # Default model
)

# 단일 문서 분류
text = "사내 도서 구입 신청 및 독서실 이용 안내"
category = classifier.classify(text)
print(category)  # Output: 지원 제도

# 배치 분류
texts = [
    "재택근무 정책 및 출퇴근 시간 규정",
    "1 on 1 미팅 가이드 및 피드백 방법",
    "온라인 강의 수강 지원 프로그램",
]
categories = classifier.classify_batch(texts)
for text, category in zip(texts, categories):
    print(f"{text} → {category}")
```

### 방법 3: 고급 설정

```python
# 예시 없이 분류 (Zero-shot)
category = classifier.classify(text, use_examples=False)

# Temperature 조정 (더 창의적인 응답)
category = classifier.classify(text, temperature=0.3)

# 다른 OpenAI 모델 사용
classifier = DocumentClassifier(model="gpt-4o")
```

## 예제 실행

### 기본 예제

```bash
# 가상환경 활성화
source .venv/bin/activate

# 예제 스크립트 실행 (마크다운 형식의 샘플 문서 분류)
python examples/classify_example.py
```

### CSV 데이터 평가

실제 CSV 데이터의 page_content를 사용하여 분류하고 정답(categoryL1)과 비교:

```bash
# 단일 문서 테스트 (인터랙티브)
python examples/test_single_document.py

# 전체 데이터셋 평가 (10개 샘플)
python examples/evaluate_classifier.py
```

**test_single_document.py 사용 예시:**

```python
# CSV에서 특정 문서를 불러와 분류하고 정답과 비교
import pandas as pd
from flex_ml.document_classifier import DocumentClassifier
from flex_ml.utils.path import RAW_DATA_PATH

# 데이터 로드
df = pd.read_csv(f"{RAW_DATA_PATH}/people_intelligence_documents.csv")

# 분류기 초기화
classifier = DocumentClassifier()

# 첫 번째 문서 테스트
row = df.iloc[0]
predicted = classifier.classify(row['page_content'])

print(f"실제 카테고리: {row['categoryL1']}")
print(f"예측 카테고리: {predicted}")
print(f"결과: {'✓ 정답' if predicted == row['categoryL1'] else '✗ 오답'}")
```

**evaluate_classifier.py 결과 예시:**

```
EVALUATION RESULTS
================================================================================

Overall Accuracy: 92.00%
Correct: 46/50

--------------------------------------------------------------------------------
Per-Category Performance:
--------------------------------------------------------------------------------
지원 제도                      | Accuracy: 95.00% (19/20)
조직원칙 및 리더십              | Accuracy: 90.91% (10/11)
근무환경 및 제도                | Accuracy: 83.33% (5/6)
구성원 여정                    | Accuracy: 100.00% (8/8)
성장 및 발전                   | Accuracy: 80.00% (4/5)
```

## API 레퍼런스

### `DocumentClassifier` 클래스

#### `__init__(api_key=None, model="gpt-4o-mini", data_path=None)`

**Parameters:**
- `api_key` (str, optional): OpenAI API 키. 미제공시 환경변수에서 로드
- `model` (str): 사용할 OpenAI 모델 (기본값: "gpt-4o-mini")
- `data_path` (str, optional): CSV 데이터 파일 경로

#### `classify(text, use_examples=True, temperature=0.0, max_tokens=50)`

단일 텍스트 분류

**Parameters:**
- `text` (str): 분류할 텍스트
- `use_examples` (bool): Few-shot 예시 사용 여부 (기본값: True)
- `temperature` (float): OpenAI temperature 파라미터 (기본값: 0.0)
- `max_tokens` (int): 최대 응답 토큰 수 (기본값: 50)

**Returns:**
- `str`: 분류된 카테고리 이름

#### `classify_batch(texts, use_examples=True, **kwargs)`

여러 텍스트 일괄 분류

**Parameters:**
- `texts` (List[str]): 분류할 텍스트 리스트
- `use_examples` (bool): Few-shot 예시 사용 여부
- `**kwargs`: `classify()` 함수에 전달할 추가 인자

**Returns:**
- `List[str]`: 분류된 카테고리 이름 리스트

### `classify_document()` 함수

간편한 단일 문서 분류 함수

```python
classify_document(text, api_key=None, model="gpt-4o-mini", use_examples=True)
```

## 데이터 구조

분류기는 다음 CSV 데이터를 활용합니다:

```
data/raw/people_intelligence_documents.csv
```

**컬럼:**
- `page_id`: 문서 고유 ID
- `categoryL1`: 대분류 (분류 타겟)
- `categoryL2`: 중분류
- `title`: 문서 제목
- `page_content`: 문서 내용

## 프롬프트 엔지니어링 전략

1. **역할 정의**: "회사 내부 문서를 분류하는 전문가"로 역할 부여
2. **명확한 카테고리 정의**: 각 카테고리에 대한 설명 제공
3. **Few-shot Learning**: 각 카테고리별 2개의 실제 예시 포함
4. **구조화된 출력**: 카테고리 이름만 출력하도록 지시
5. **Deterministic Output**: Temperature 0.0으로 일관성 확보

## 성능 최적화

- **모델 선택**: `gpt-4o-mini`는 비용 효율적이면서 우수한 성능
- **Few-shot Examples**: 분류 정확도 향상
- **배치 처리**: 여러 문서를 효율적으로 처리

## 주의사항

1. OpenAI API 키가 필요합니다
2. API 호출 비용이 발생합니다
3. 네트워크 연결이 필요합니다
4. 응답 시간은 네트워크 상태에 따라 달라집니다

## 라이선스

이 프로젝트는 Flex AI 내부용입니다.