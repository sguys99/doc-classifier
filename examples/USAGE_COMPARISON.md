# Query Generation 사용법 비교

HR 질의응답 챗봇을 위한 query 생성 방법을 비교합니다.

## 빠른 선택 가이드

| 필요사항 | 사용 스크립트 | 출력 형태 |
|---------|------------|----------|
| **문서당 1개 질문 (문자열)** | `generate_single_query_per_row.py` | `'질문'` |
| **문서당 여러 질문 (리스트)** | `generate_hr_queries.py` | `['질문1', '질문2', '질문3']` |
| **대량 처리 (1개씩)** | `generate_single_query_batch.py` | `'질문'` |
| **대량 처리 (여러 개씩)** | `generate_hr_queries_batch.py` | `['질문1', '질문2', '질문3']` |

## 상세 비교

### Option 1: 문서당 1개 질문 생성 (Single Query)

**언제 사용?**
- RAG 시스템에서 1:1 매핑이 필요할 때
- 각 문서의 대표 질문만 필요할 때
- 데이터 구조를 단순하게 유지하고 싶을 때

**스크립트:**
```bash
# 샘플 (5개 문서)
python examples/generate_single_query_per_row.py

# 전체 문서 배치 처리
python examples/generate_single_query_batch.py --batch-size 10
```

**Python 코드:**
```python
from examples.generate_single_query_per_row import generate_single_query_per_row

df_with_query = generate_single_query_per_row(df)
# query 컬럼: '질문' (문자열)
```

**출력 예시:**
```csv
page_id,title,query
abc-123,Device Trust 가이드,Device Trust 정책은 언제부터 시행되나요?
```

**자세한 가이드:** [README_single_query.md](README_single_query.md)

---

### Option 2: 문서당 여러 질문 생성 (Multiple Queries)

**언제 사용?**
- 다양한 관점의 질문이 필요할 때
- 챗봇 학습 데이터의 다양성을 높이고 싶을 때
- 각 문서에 대한 다양한 유형의 질문이 필요할 때

**스크립트:**
```bash
# 샘플 (5개 문서, 각 3개 질문)
python examples/generate_hr_queries.py

# 전체 문서 배치 처리 (각 3개 질문)
python examples/generate_hr_queries_batch.py --num-queries 3
```

**Python 코드:**
```python
from flex_ml.models import generate_queries

df_with_queries = generate_queries(df, num_queries=3)
# query 컬럼: ['질문1', '질문2', '질문3'] (리스트)
```

**출력 예시:**
```csv
page_id,title,query
abc-123,Device Trust 가이드,"['질문1', '질문2', '질문3']"
```

**자세한 가이드:** [README_query_generation.md](README_query_generation.md)

---

## 실행 예시 비교

### Single Query (1개)

```bash
$ python examples/generate_single_query_per_row.py

Generated Query:
  Device Trust 정책은 언제부터 시행되나요?

Type of query column: <class 'str'>
```

### Multiple Queries (3개)

```bash
$ python examples/generate_hr_queries.py

Generated Queries:
  1. Device Trust 정책은 언제부터 시행되나요?
  2. Okta Verify 앱 문제가 발생하면 어디에 문의하나요?
  3. Windows에서 지문 인식을 어떻게 설정하나요?

Type of query column: <class 'list'>
```

---

## 비용 및 성능 비교 (89개 문서 기준)

| 방식 | Query 수 | 예상 비용 | 처리 시간 |
|------|---------|----------|---------|
| **Single (1개)** | 89개 | $0.05-0.20 | 3-5분 |
| **Multiple (3개)** | 267개 | $0.10-0.50 | 8-15분 |
| **Multiple (5개)** | 445개 | $0.15-0.80 | 12-20분 |

*gpt-4o-mini 기준

---

## 배치 처리 옵션 비교

### Single Query 배치

```bash
python examples/generate_single_query_batch.py \
  --batch-size 10 \
  --temperature 0.7 \
  --limit 50
```

### Multiple Queries 배치

```bash
python examples/generate_hr_queries_batch.py \
  --batch-size 10 \
  --num-queries 3 \
  --temperature 0.7 \
  --limit 50
```

**공통 옵션:**
- `--batch-size`: 배치 크기
- `--temperature`: 생성 창의성 (0.0-1.0)
- `--start-from`: 시작 인덱스 (재개용)
- `--limit`: 처리할 문서 수 제한
- `--output-name`: 출력 파일명

**차이점:**
- Multiple Queries만 `--num-queries` 옵션 제공

---

## CSV 출력 형식 비교

### Single Query
```csv
page_id,categoryL1,categoryL2,title,page_content,query
abc-123,지원 제도,업무 지원,Device Trust 가이드,"내용...",Device Trust 정책은 언제부터 시행되나요?
```

### Multiple Queries
```csv
page_id,categoryL1,categoryL2,title,page_content,query
abc-123,지원 제도,업무 지원,Device Trust 가이드,"내용...","['질문1', '질문2', '질문3']"
```

---

## 다음 단계별 활용 방법

### 1. RAG 시스템 구축
```python
# Single Query 사용 권장
from examples.generate_single_query_per_row import generate_single_query_per_row

df = generate_single_query_per_row(df)

# Embedding 생성
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
query_embeddings = model.encode(df['query'].tolist())
```

### 2. 챗봇 Fine-tuning
```python
# Multiple Queries 사용 권장 (데이터 다양성)
from flex_ml.models import generate_queries

df = generate_queries(df, num_queries=5)

# 리스트를 펼쳐서 학습 데이터 생성
training_data = []
for _, row in df.iterrows():
    for query in row['query']:
        training_data.append({
            'query': query,
            'document': row['page_content'],
            'category': row['categoryL1']
        })
```

### 3. 검색 평가 데이터셋
```python
# Single Query 사용 권장 (1:1 평가)
df = generate_single_query_per_row(df)

# 평가 데이터셋 생성
eval_data = df[['query', 'page_id', 'page_content']].rename(
    columns={'query': 'question', 'page_content': 'expected_answer'}
)
```

---

## 추천 사용 시나리오

### 🎯 Single Query를 사용하세요

1. **RAG 시스템 구축**
   - 문서-질문 1:1 매핑
   - Semantic search 기반 검색

2. **데이터 구조가 중요한 경우**
   - CSV를 다른 시스템과 통합
   - 데이터베이스 저장 (리스트 처리 복잡도 회피)

3. **비용/시간 절약**
   - 빠른 프로토타이핑
   - 제한된 예산

### 🎯 Multiple Queries를 사용하세요

1. **챗봇 학습 데이터 생성**
   - Fine-tuning용 데이터
   - 다양한 질문 패턴 학습

2. **평가 데이터셋 구축**
   - 다양한 각도의 테스트 케이스
   - 강건성 평가

3. **풍부한 데이터가 필요한 경우**
   - 충분한 예산과 시간
   - 고품질 학습 데이터 요구사항

---

## 빠른 시작

### 1분 안에 시작하기

```bash
# 환경 설정
export OPENAI_API_KEY="your-key-here"
source .venv/bin/activate

# Single Query 생성 (빠르고 간단)
python examples/generate_single_query_per_row.py

# 또는 Multiple Queries 생성 (다양한 질문)
python examples/generate_hr_queries.py
```

### 전체 데이터 처리하기

```bash
# Single Query (권장: 간단한 구조)
python examples/generate_single_query_batch.py

# Multiple Queries (권장: 풍부한 데이터)
python examples/generate_hr_queries_batch.py
```

---

## 문제 해결

### "리스트를 문자열로 변환하고 싶어요"

```python
# Multiple Queries 결과를 Single Query로 변환
df['query'] = df['query'].apply(lambda x: x[0] if isinstance(x, list) else x)
```

### "문자열을 리스트로 변환하고 싶어요"

```python
# Single Query 결과를 Multiple Queries로 변환
df['query'] = df['query'].apply(lambda x: [x] if isinstance(x, str) else x)
```

---

## 참고 문서

- **Single Query 가이드**: [README_single_query.md](README_single_query.md)
- **Multiple Queries 가이드**: [README_query_generation.md](README_query_generation.md)
- **QueryGenerator 소스코드**: [../src/flex_ml/models/query_generator.py](../src/flex_ml/models/query_generator.py)