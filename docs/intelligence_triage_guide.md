# Intelligence Triage Guide

## Overview

Intelligence Triage is a feature that classifies user queries into predefined categories using OpenAI's LLM with few-shot prompting. This helps route user questions to the appropriate category for better information retrieval and support.

## Features

- **Two-Level Classification**:
  - **L1**: 6 broad categories (지원 제도, 조직원칙 및 리더십, 근무환경 및 제도, 구성원 여정, 성장 및 발전, 기타)
  - **L2**: 13 detailed categories (업무 지원, 생활 지원, 리더십, 문화/ 팀빌딩, etc.)

- **Few-Shot Prompting**: Uses examples from the training data to improve classification accuracy

- **Structured Output**: Uses Pydantic models to ensure valid category outputs

- **Flexible API**: Supports both single query and batch classification

## Quick Start

### 1. Set up your environment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Activate virtual environment
source .venv/bin/activate
```

### 2. Basic Usage

```python
from flex_ml.models import IntelligenceTriage, triage_query

# Simple classification (L1)
category = triage_query("입사 지원서는 어떻게 제출하나요?")
print(category)  # Output: 구성원 여정

# Detailed classification (L2)
category = triage_query("입사 지원서는 어떻게 제출하나요?", classification_level="L2")
print(category)  # Output: 채용
```

### 3. Advanced Usage

```python
from flex_ml.models import IntelligenceTriage

# Initialize the classifier
triage = IntelligenceTriage(
    model="gpt-4o-mini",  # OpenAI model to use
    classification_level="L1",  # or "L2"
    api_key=None,  # Uses OPENAI_API_KEY env var if None
)

# Classify a single query
query = "Device Trust 인증을 설정하는 방법을 알려주세요"
category = triage.classify(query)
print(f"Category: {category}")  # Output: 지원 제도

# Classify multiple queries
queries = [
    "도서 구매는 어떻게 신청하나요?",
    "휴가를 사용하려면 어떻게 해야 하나요?",
    "리크루팅 식사 비용은 어떻게 청구하나요?",
]
categories = triage.classify_batch(queries)
for q, c in zip(queries, categories):
    print(f"{q} -> {c}")
```

### 4. Without Few-Shot Examples

```python
# Classify without using few-shot examples
category = triage.classify(query, use_examples=False)
```

## Category Definitions

### L1 Categories (6 categories)

1. **지원 제도** - Support systems for work and life
2. **조직원칙 및 리더십** - Organizational principles and leadership
3. **근무환경 및 제도** - Work environment and policies
4. **구성원 여정** - Member journey (hiring to offboarding)
5. **성장 및 발전** - Growth and development
6. **기타** - Other

### L2 Categories (13 categories)

1. **업무 지원** - Work support (IT, devices, etc.)
2. **생활 지원** - Life support (fitness, health, etc.)
3. **리더십** - Leadership (1on1, coaching, etc.)
4. **문화/ 팀빌딩** - Culture and team building
5. **원칙/ 철학** - Principles and philosophy
6. **협업 방식** - Collaboration methods
7. **근무 제도** - Work policies (hours, leave, etc.)
8. **오피스** - Office facilities
9. **채용** - Recruitment
10. **온보딩** - Onboarding
11. **오프보딩** - Offboarding
12. **구성원을 위한 개인정보 처리방침** - Privacy policy for members
13. **성과/성장** - Performance and growth

## Configuration

The classifier uses YAML configuration files for prompts:

- **L1 Prompt**: `configs/triage_prompt_l1.yaml`
- **L2 Prompt**: `configs/triage_prompt_l2.yaml`

You can customize these files to adjust the classification behavior.

## Training Data

The classifier uses few-shot examples from:
- `data/raw/people_intelligence_documents_with_queries.csv`

This file contains queries with their corresponding L1 and L2 categories.

## Examples

Run the complete example script:

```bash
python examples/intelligence_triage_example.py
```

This will demonstrate:
- L1 classification
- L2 classification
- Convenience function usage
- Batch classification
- Classification with/without few-shot examples

## Evaluation

Evaluate the classifier's performance on the actual query dataset:

```bash
# Evaluate L1 classification on all queries (89 queries)
python examples/evaluate_intelligence_triage.py --level L1

# Evaluate L2 classification on 20 samples
python examples/evaluate_intelligence_triage.py --level L2 --sample-size 20

# Evaluate without few-shot examples
python examples/evaluate_intelligence_triage.py --level L1 --no-examples
```

The evaluation script will:
- Load queries from `data/raw/people_intelligence_documents_with_queries.csv`
- Classify each query using the IntelligenceTriage classifier
- Compare predictions with actual category labels
- Compute overall and per-category accuracy
- Display confusion matrix and misclassified queries
- Save results to `data/intermediate/triage_evaluation_results_[l1|l2].csv`

**Example Output:**
```
Overall Accuracy: 85.00%
Correct: 17/20

Per-Category Performance:
지원 제도                               | Accuracy: 90.00% (9/10)
조직원칙 및 리더십                          | Accuracy: 75.00% (3/4)
근무환경 및 제도                           | Accuracy: 100.00% (3/3)
...
```

## API Reference

### `IntelligenceTriage`

Main class for query classification.

**Parameters:**
- `api_key` (Optional[str]): OpenAI API key. Defaults to `OPENAI_API_KEY` env var.
- `model` (str): OpenAI model name. Default: "gpt-4o-mini"
- `data_path` (Optional[str]): Path to CSV with training examples
- `classification_level` (Literal["L1", "L2"]): Classification level
- `prompt_config_path` (Optional[str]): Path to custom prompt config

**Methods:**
- `classify(query, use_examples=True, temperature=0.0)`: Classify a single query
- `classify_batch(queries, use_examples=True, **kwargs)`: Classify multiple queries

### `triage_query()`

Convenience function for quick classification.

**Parameters:**
- `query` (str): User query to classify
- `api_key` (Optional[str]): OpenAI API key
- `model` (str): OpenAI model. Default: "gpt-4o-mini"
- `use_examples` (bool): Use few-shot examples. Default: True
- `classification_level` (Literal["L1", "L2"]): Classification level. Default: "L1"

**Returns:**
- `str`: Classified category name

## Best Practices

1. **Use L1 for broad categorization**: Start with L1 if you need general routing
2. **Use L2 for detailed routing**: Use L2 when you need fine-grained categorization
3. **Enable few-shot examples**: Keep `use_examples=True` for better accuracy
4. **Set temperature to 0**: Use `temperature=0.0` for consistent results
5. **Batch processing**: Use `classify_batch()` for multiple queries to optimize API calls

## Performance Tips

- **Caching**: Consider caching results for frequently asked queries
- **Batch API calls**: Use `classify_batch()` instead of multiple `classify()` calls
- **Model selection**: Use "gpt-4o-mini" for cost-effective classification, "gpt-4o" for higher accuracy

## Troubleshooting

### "OpenAI API key is required" error

Make sure to set the environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Classification seems inaccurate

1. Check if few-shot examples are enabled: `use_examples=True`
2. Review and update prompt configuration files
3. Ensure training data quality in the CSV file
4. Try using a more powerful model like "gpt-4o"

### File not found errors

Make sure you're running from the repository root directory:
```bash
cd /path/to/doc-classifier
python examples/intelligence_triage_example.py
```
