# Evaluation Guide

DocumentClassifier 평가 스크립트 사용 가이드입니다.

## 개요

`evaluate_classifier.py` 스크립트는 CSV 데이터를 사용하여 분류기의 성능을 평가합니다.

- L1 (6개 카테고리) 또는 L2 (13개 카테고리) 분류 평가 지원
- 전체 데이터셋 또는 샘플링된 데이터로 평가 가능
- Few-shot learning 사용/미사용 선택 가능
- 상세한 성능 리포트 및 혼동 행렬(confusion matrix) 생성

## 사용 방법

### 기본 사용법

```bash
# 환경 변수 설정
export OPENAI_API_KEY="your-api-key-here"

# 가상환경 활성화
source .venv/bin/activate

# L1 분류 평가 (전체 데이터)
python examples/evaluate_classifier.py --level L1

# L2 분류 평가 (전체 데이터)
python examples/evaluate_classifier.py --level L2
```

### 샘플링을 사용한 평가

```bash
# L1 분류: 20개 샘플로만 평가
python examples/evaluate_classifier.py --level L1 --sample-size 20

# L2 분류: 50개 샘플로만 평가
python examples/evaluate_classifier.py --level L2 --sample-size 50
```

### Few-shot Examples 없이 평가

```bash
# Few-shot examples를 사용하지 않고 평가
python examples/evaluate_classifier.py --level L1 --no-examples

# L2 분류, 샘플링 + few-shot 없이
python examples/evaluate_classifier.py --level L2 --sample-size 30 --no-examples
```

### 커스텀 랜덤 시드

```bash
# 다른 샘플링 결과를 위해 시드 변경
python examples/evaluate_classifier.py --level L1 --sample-size 20 --seed 123
```

## 명령행 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--level` | str | `L1` | 분류 수준 (`L1` 또는 `L2`) |
| `--sample-size` | int | `None` | 평가할 샘플 수 (기본값: 전체 데이터) |
| `--no-examples` | flag | `False` | Few-shot examples 비활성화 |
| `--seed` | int | `42` | 샘플링을 위한 랜덤 시드 |

### 도움말 보기

```bash
python examples/evaluate_classifier.py --help
```

## 출력 결과

### 1. 평가 진행 중
```
Total documents in dataset: 89
Classification level: L1
Evaluating on all 89 documents

Initializing classifier...

Classifying documents...
[1/89] Actual: 지원 제도 | Predicted: 지원 제도 | ✓
[2/89] Actual: 조직원칙 및 리더십 | Predicted: 조직원칙 및 리더십 | ✓
...
```

### 2. 평가 결과 리포트

```
================================================================================
EVALUATION RESULTS
================================================================================

Overall Accuracy: 85.39%
Correct: 76/89

--------------------------------------------------------------------------------
Per-Category Performance:
--------------------------------------------------------------------------------
지원 제도                      | Accuracy: 90.00% (36/40)
조직원칙 및 리더십              | Accuracy: 81.82% (18/22)
근무환경 및 제도                | Accuracy: 83.33% (10/12)
구성원 여정                    | Accuracy: 80.00% (8/10)
성장 및 발전                    | Accuracy: 80.00% (4/5)
기타                          | Accuracy: 0.00% (0/0)

--------------------------------------------------------------------------------
Misclassified Documents:
--------------------------------------------------------------------------------

Title: 1 on 1 미팅 가이드
  Actual:    조직원칙 및 리더십
  Predicted: 성장 및 발전
...

--------------------------------------------------------------------------------
Confusion Matrix:
--------------------------------------------------------------------------------
```

### 3. 결과 파일 저장

평가 결과는 자동으로 CSV 파일로 저장됩니다:

- L1 평가: `data/intermediate/evaluation_results_l1.csv`
- L2 평가: `data/intermediate/evaluation_results_l2.csv`

**CSV 파일 구조:**
```csv
page_id,title,actual,predicted,correct
1,문서 제목,지원 제도,지원 제도,True
2,문서 제목,조직원칙 및 리더십,조직원칙 및 리더십,True
...
```

## 평가 지표

### 전체 정확도 (Overall Accuracy)
전체 문서 중 올바르게 분류된 문서의 비율

### 카테고리별 정확도 (Per-Category Accuracy)
각 카테고리별로:
- 해당 카테고리에 속한 문서 수
- 올바르게 분류된 문서 수
- 카테고리별 정확도

### 혼동 행렬 (Confusion Matrix)
실제 카테고리 vs 예측 카테고리의 교차표

## 평가 시 고려사항

### 샘플 크기 선택
- **전체 데이터 평가**: 가장 정확한 성능 측정, 하지만 시간과 비용이 많이 듦
- **샘플링 평가**: 빠른 테스트, 대략적인 성능 파악 (권장: 20-50개)

### Few-shot Examples 사용
- **사용 (기본값)**: 일반적으로 더 높은 정확도
- **미사용 (`--no-examples`)**: 프롬프트만으로 성능 측정, 속도는 빠름

### L1 vs L2 평가
- **L1**: 더 넓은 카테고리, 일반적으로 더 높은 정확도 예상
- **L2**: 더 세밀한 분류, 정확도는 다소 낮을 수 있음

## 예시 워크플로우

### 1. 빠른 테스트 (개발 중)
```bash
# 10개 샘플로 빠르게 테스트
python examples/evaluate_classifier.py --level L1 --sample-size 10
```

### 2. 중간 규모 평가 (프롬프트 튜닝 시)
```bash
# 30개 샘플로 평가
python examples/evaluate_classifier.py --level L1 --sample-size 30
python examples/evaluate_classifier.py --level L2 --sample-size 30
```

### 3. 최종 성능 평가 (배포 전)
```bash
# 전체 데이터로 평가
python examples/evaluate_classifier.py --level L1
python examples/evaluate_classifier.py --level L2
```

## 성능 개선 팁

1. **오분류 문서 분석**: 리포트의 "Misclassified Documents" 섹션을 확인하여 패턴 파악
2. **프롬프트 튜닝**: `configs/classifier_prompt_l1.yaml` 또는 `classifier_prompt_l2.yaml` 수정
3. **Few-shot 예시 조정**: `DocumentClassifier._prepare_examples()` 메서드에서 예시 개수 변경
4. **모델 변경**: `DocumentClassifier` 초기화 시 다른 모델 사용 (예: `gpt-4o`)

## 문제 해결

### API 키 오류
```
⚠️  OPENAI_API_KEY environment variable is not set.
```
→ `export OPENAI_API_KEY="your-key"` 실행

### 메모리 부족
→ `--sample-size` 옵션으로 평가 문서 수 제한

### 평가 속도가 너무 느림
→ `--sample-size` 사용 또는 `--no-examples` 옵션 추가
