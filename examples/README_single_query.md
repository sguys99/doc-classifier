# HR Single Query Generation Guide

이 가이드는 각 문서(row)당 **하나의 질문만** 생성하는 방법을 설명합니다.

## 개요

기본 `generate_queries` 함수는 각 문서당 여러 개의 질문을 생성하여 리스트로 반환합니다.
하지만 때로는 각 문서당 **하나의 대표 질문만** 필요한 경우가 있습니다.

이 가이드에서는 각 row에 **단일 문자열(string)** 형태의 query를 생성하는 방법을 제공합니다.

## 출력 형식 비교

### 기존 방식 (여러 query를 리스트로)

```python
from flex_ml.models import generate_queries

df_with_queries = generate_queries(df, num_queries=3)
# query 컬럼: ['질문1', '질문2', '질문3']  <- 리스트
```

| page_id | title | query |
|---------|-------|-------|
| abc-123 | Device Trust | ['질문1', '질문2', '질문3'] |

### 새로운 방식 (하나의 query를 문자열로)

```python
from examples.generate_single_query_per_row import generate_single_query_per_row

df_with_query = generate_single_query_per_row(df)
# query 컬럼: '질문1'  <- 문자열
```

| page_id | title | query |
|---------|-------|-------|
| abc-123 | Device Trust | 질문1 |

## 사용 방법

### 방법 1: 간단한 예제 스크립트 사용

```bash
# 환경 활성화
source .venv/bin/activate

# 5개 샘플 문서에 대해 1개씩 query 생성
python examples/generate_single_query_per_row.py
```

이 스크립트는:
- 5개 샘플 문서에 대해 각각 1개의 query 생성
- 결과를 `data/processed/sample_single_query_per_row.csv`에 저장
- 각 query는 문자열로 저장됨

### 방법 2: 배치 처리 스크립트 사용

```bash
# 전체 문서 처리 (10개씩 배치)
python examples/generate_single_query_batch.py

# 옵션 사용 예시
python examples/generate_single_query_batch.py \
  --batch-size 20 \
  --temperature 0.8 \
  --limit 50 \
  --output-name my_queries.csv
```

#### 배치 스크립트 옵션

- `--batch-size`: 배치 크기 (기본값: 10)
- `--temperature`: 생성 창의성 (0.0-1.0, 기본값: 0.7)
- `--start-from`: 시작 인덱스 (재개용, 기본값: 0)
- `--limit`: 처리할 문서 수 제한 (기본값: 전체)
- `--output-name`: 출력 파일명 (기본값: 타임스탬프 포함 자동 생성)

### 방법 3: Python 코드에서 직접 사용

```python
import pandas as pd
from flex_ml.models import QueryGenerator
from flex_ml.utils.path import RAW_DATA_PATH
import os

# 데이터 로드
df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'people_intelligence_documents.csv'))

# 1개 query만 생성하도록 설정
generator = QueryGenerator(num_queries=1)

# 각 문서에 대해 query 생성
queries = []
for idx, row in df.iterrows():
    query_list = generator.generate(
        page_content=row['page_content'],
        title=row['title'],
    )
    # 리스트에서 첫 번째(유일한) query 추출
    single_query = query_list[0]
    queries.append(single_query)

# DataFrame에 추가
df['query'] = queries

# 저장
df.to_csv('output.csv', index=False)
```

### 방법 4: 함수로 래핑하여 사용

```python
from examples.generate_single_query_per_row import generate_single_query_per_row
import pandas as pd

# 데이터 로드
df = pd.read_csv('data/raw/people_intelligence_documents.csv')

# 단일 query 생성
df_with_query = generate_single_query_per_row(df, temperature=0.7)

# 저장
df_with_query.to_csv('output.csv', index=False)
```

## 실행 예시

```bash
$ python examples/generate_single_query_per_row.py

Loading data from: /Users/.../data/raw/people_intelligence_documents.csv
Loaded 89 documents

Generating single query for 5 sample documents...
Generating query for document 1/5...
Generating query for document 2/5...
Generating query for document 3/5...
Generating query for document 4/5...
Generating query for document 5/5...

================================================================================
SAMPLE RESULTS (1 query per row)
================================================================================

Document 1:
Title: [IT] Device Trust (인증, 등록) 가이드
Category L1: 지원 제도
Category L2: 업무 지원
Content preview: ...

Generated Query:
  2024년 10월 10일 이후에 Device Trust 정책이 시행되면, 어떤 애플리케이션에 접근할 수 없게 되나요?
--------------------------------------------------------------------------------
...

Sample results saved to: /Users/.../data/processed/sample_single_query_per_row.csv

================================================================================
DATA STRUCTURE
================================================================================
Type of query column: <class 'str'>
Sample query value: '2024년 10월 10일 이후에 Device Trust 정책이 시행되면, 어떤 애플리케이션에 접근할 수 없게 되나요?'
```

## 출력 CSV 형식

생성된 CSV 파일의 구조:

```csv
page_id,categoryL1,categoryL2,title,page_content,query
abc-123,지원 제도,업무 지원,[IT] Device Trust 가이드,"문서 내용...","Device Trust 정책은 언제부터 시행되나요?"
def-456,근무환경 및 제도,복지,"복지 제도 안내","문서 내용...","회사에서 제공하는 복지 혜택은 무엇이 있나요?"
```

**핵심 포인트:**
- `query` 컬럼은 **문자열(string)** 타입입니다
- 각 row당 정확히 **1개의 질문만** 포함됩니다
- 리스트나 JSON이 아닌 **순수 텍스트**로 저장됩니다

## 비용 및 성능

### 비용 (gpt-4o-mini 기준)

- 문서당 1개 query 생성
- 89개 문서 전체: 약 $0.05-0.20

### 처리 시간

- 문서당 약 2-3초
- 89개 전체: 약 3-5분

## 언제 사용하나요?

### Single Query (1개) 사용 시나리오

- ✅ RAG 시스템에서 문서-질문 매핑이 1:1인 경우
- ✅ 각 문서의 **대표 질문** 1개만 필요한 경우
- ✅ 데이터베이스나 검색 시스템에서 간단한 구조를 원할 때
- ✅ CSV를 다른 시스템과 통합할 때 (리스트 형식 처리가 어려운 경우)

### Multiple Queries (여러 개) 사용 시나리오

- ✅ 다양한 관점의 질문을 생성하고 싶을 때
- ✅ 챗봇 학습 데이터의 다양성을 높이고 싶을 때
- ✅ 각 문서에 대해 여러 유형의 질문 (절차, 자격, 기한 등)이 필요할 때

## 다음 단계

단일 query를 생성한 후:

1. **RAG 시스템 구축**
   ```python
   # query를 사용하여 semantic search
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   query_embeddings = model.encode(df['query'].tolist())
   ```

2. **Q&A 데이터셋 생성**
   ```python
   # query-answer pair 생성
   qa_dataset = df[['query', 'page_content']].rename(
       columns={'query': 'question', 'page_content': 'answer'}
   )
   ```

3. **검색 시스템 평가**
   ```python
   # 생성된 query로 검색 성능 테스트
   from sklearn.metrics.pairwise import cosine_similarity

   # query embedding vs document embedding
   similarities = cosine_similarity(query_embeddings, doc_embeddings)
   ```

## 문제 해결

### "Type of query column: list" 오류

기존 `generate_queries` 함수를 사용하면 리스트가 반환됩니다.
대신 `generate_single_query_per_row` 함수를 사용하세요.

```python
# ❌ 이렇게 하면 리스트가 반환됨
from flex_ml.models import generate_queries
df = generate_queries(df, num_queries=1)  # 여전히 ['query'] 형태

# ✅ 이렇게 하면 문자열이 반환됨
from examples.generate_single_query_per_row import generate_single_query_per_row
df = generate_single_query_per_row(df)  # 'query' 형태
```

### CSV 읽을 때 query가 리스트로 파싱되는 문제

```python
# 문자열로 강제 변환
df = pd.read_csv('output.csv')
df['query'] = df['query'].astype(str)
```

## 참고 자료

- 여러 query 생성: [README_query_generation.md](README_query_generation.md)
- 문서 분류: 프로젝트 메인 README
- QueryGenerator 클래스: [src/flex_ml/models/query_generator.py](../src/flex_ml/models/query_generator.py)