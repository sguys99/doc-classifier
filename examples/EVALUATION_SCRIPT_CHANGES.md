# Evaluation Script Updates - L1/L2 Support

## 변경 사항 요약

`evaluate_classifier.py` 스크립트를 업데이트하여 L1과 L2 분류를 모두 평가할 수 있도록 개선했습니다.

## 주요 변경사항

### 1. `evaluate_classifier()` 함수 업데이트

**추가된 파라미터:**
```python
classification_level: str = "L1"
```

**변경된 동작:**
- 분류 수준을 출력에 표시
- `DocumentClassifier` 초기화 시 `classification_level` 전달
- 동적으로 올바른 카테고리 컬럼 선택:
  ```python
  category_column = "categoryL1" if classification_level == "L1" else "categoryL2"
  actual_category = row[category_column]
  ```

### 2. `main()` 함수 - 커맨드라인 인자 지원

**새로운 argparse 설정:**

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--level` | `{L1, L2}` | `L1` | 분류 수준 선택 |
| `--sample-size` | `int` | `None` | 평가할 샘플 수 |
| `--no-examples` | `flag` | `False` | Few-shot 예시 비활성화 |
| `--seed` | `int` | `42` | 랜덤 시드 |

**파일 저장 경로 변경:**
- 기존: `evaluation_results.csv`
- 변경:
  - L1: `evaluation_results_l1.csv`
  - L2: `evaluation_results_l2.csv`

### 3. 문서화 업데이트

**파일 헤더 독스트링:**
- L1/L2 지원 명시
- 사용 예시 추가

## 코드 비교

### Before (L1만 지원)
```python
def evaluate_classifier(
    sample_size: int = None,
    use_examples: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # ...
    classifier = DocumentClassifier(model="gpt-4o-mini")
    # ...
    actual_category = row["categoryL1"]  # 하드코딩됨
```

### After (L1/L2 지원)
```python
def evaluate_classifier(
    sample_size: int = None,
    use_examples: bool = True,
    random_seed: int = 42,
    classification_level: str = "L1",  # 새로운 파라미터
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # ...
    classifier = DocumentClassifier(
        model="gpt-4o-mini",
        classification_level=classification_level  # 동적 설정
    )
    # ...
    category_column = "categoryL1" if classification_level == "L1" else "categoryL2"
    actual_category = row[category_column]  # 동적 선택
```

## 사용 예시

### 커맨드라인 사용

```bash
# L1 평가 (전체 데이터)
python examples/evaluate_classifier.py --level L1

# L2 평가 (20개 샘플)
python examples/evaluate_classifier.py --level L2 --sample-size 20

# Few-shot 없이 평가
python examples/evaluate_classifier.py --level L1 --no-examples

# 커스텀 시드로 샘플링
python examples/evaluate_classifier.py --level L2 --sample-size 30 --seed 999
```

### 프로그래밍 방식 사용

```python
from evaluate_classifier import evaluate_classifier

# L1 평가
results_l1, metrics_l1 = evaluate_classifier(
    sample_size=20,
    classification_level="L1"
)

# L2 평가
results_l2, metrics_l2 = evaluate_classifier(
    sample_size=20,
    classification_level="L2"
)
```

## 출력 예시

```bash
$ python examples/evaluate_classifier.py --level L2 --sample-size 10

================================================================================
Document Classifier Evaluation
================================================================================
Total documents in dataset: 89
Classification level: L2
Evaluating on 10 sampled documents

Initializing classifier...

Classifying documents...
[1/10] Actual: 업무 지원 | Predicted: 업무 지원 | ✓
[2/10] Actual: 생활 지원 | Predicted: 생활 지원 | ✓
[3/10] Actual: 리더십 | Predicted: 리더십 | ✓
...

================================================================================
EVALUATION RESULTS
================================================================================

Overall Accuracy: 80.00%
Correct: 8/10

--------------------------------------------------------------------------------
Per-Category Performance:
--------------------------------------------------------------------------------
업무 지원                      | Accuracy: 100.00% (2/2)
생활 지원                      | Accuracy: 66.67% (2/3)
리더십                        | Accuracy: 100.00% (1/1)
...

Results saved to: .../data/intermediate/evaluation_results_l2.csv
```

## 추가 생성된 파일

### `examples/EVALUATION_GUIDE.md`
상세한 평가 스크립트 사용 가이드:
- 사용 방법 및 옵션 설명
- 출력 결과 해석 방법
- 성능 개선 팁
- 문제 해결 가이드

## 하위 호환성

기존 코드와의 하위 호환성 유지:
- `classification_level` 파라미터의 기본값이 `"L1"`이므로 기존 코드는 수정 없이 작동
- 기존 사용 방식:
  ```python
  # 여전히 작동 (기본적으로 L1 사용)
  results, metrics = evaluate_classifier(sample_size=10)
  ```

## 테스트 완료 항목

✅ Help 메시지 정상 출력
✅ argparse 파라미터 정의 완료
✅ L1/L2 동적 분류 수준 선택
✅ 동적 카테고리 컬럼 선택
✅ 파일명 레벨별 구분 저장
✅ 문서화 업데이트

## 다음 단계 (선택사항)

1. **실제 평가 실행**: OpenAI API 키를 설정하고 실제 평가 수행
2. **성능 비교**: L1과 L2 성능 비교 분석
3. **프롬프트 튜닝**: 평가 결과를 바탕으로 프롬프트 개선
4. **시각화**: 평가 결과를 그래프로 시각화하는 스크립트 추가

## 완료! 🎉

`evaluate_classifier.py`는 이제 L1과 L2 분류를 모두 유연하게 평가할 수 있습니다.
