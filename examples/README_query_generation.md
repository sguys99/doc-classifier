# HR Query Generation Guide

This guide explains how to use the `generate_queries` function to create training data for an HR Q&A chatbot.

## Overview

The `generate_queries` function reads the `page_content` column from your HR documents CSV file and automatically generates relevant questions that employees might ask about that content. The generated queries are added to a new `query` column.

## Installation & Setup

1. Ensure you have the environment set up:
```bash
make init-dev
source .venv/bin/activate
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### Option 1: Use the convenience function

```python
import pandas as pd
from flex_ml.models import generate_queries

# Load your data
df = pd.read_csv("data/raw/people_intelligence_documents.csv")

# Generate queries (this will process all documents)
df_with_queries = generate_queries(
    df,
    num_queries=3,      # Generate 3 queries per document
    temperature=0.7,    # Creativity level (0.0-1.0)
)

# Save the results
df_with_queries.to_csv("data/processed/documents_with_queries.csv", index=False)
```

### Option 2: Use the QueryGenerator class for more control

```python
from flex_ml.models import QueryGenerator

# Initialize the generator
generator = QueryGenerator(
    model="gpt-4o-mini",
    num_queries=3,
)

# Generate queries for a single document
queries = generator.generate(
    page_content="문서 내용...",
    title="문서 제목",
)

print(queries)
# Output: ['질문 1', '질문 2', '질문 3']
```

## Running the Example Script

A ready-to-use example script is provided:

```bash
# Edit the script if needed
vim examples/generate_hr_queries.py

# Run it
source .venv/bin/activate
python examples/generate_hr_queries.py
```

This script will:
1. Load the HR documents from `data/raw/people_intelligence_documents.csv`
2. Generate queries for a sample of 5 documents (for testing)
3. Display the results
4. Save to `data/processed/sample_documents_with_queries.csv`

To process ALL documents, uncomment the relevant section in the script.

## Output Format

The function adds a `query` column to your DataFrame containing a list of generated queries:

| page_id | title | page_content | query |
|---------|-------|--------------|-------|
| abc-123 | Device Trust 가이드 | 문서 내용... | ['질문1', '질문2', '질문3'] |

## Configuration Options

### Parameters

- **num_queries** (int, default=3): Number of queries to generate per document
- **temperature** (float, default=0.7): Controls randomness/creativity
  - 0.0 = More deterministic
  - 1.0 = More creative
- **model** (str, default="gpt-4o-mini"): OpenAI model to use
  - "gpt-4o-mini" - Fast and cost-effective
  - "gpt-4o" - More powerful but slower

### Example with custom settings

```python
df_with_queries = generate_queries(
    df,
    num_queries=5,        # Generate more queries
    temperature=0.9,      # More creative questions
    model="gpt-4o",       # Use more powerful model
)
```

## Processing Large Datasets

For large datasets (many documents), consider processing in batches:

```python
import pandas as pd
from flex_ml.models import generate_queries

# Load data
df = pd.read_csv("data/raw/people_intelligence_documents.csv")

# Process in batches of 10
batch_size = 10
results = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}...")

    batch_with_queries = generate_queries(batch, num_queries=3)
    results.append(batch_with_queries)

    # Optional: save intermediate results
    batch_with_queries.to_csv(
        f"data/processed/batch_{i//batch_size + 1}.csv",
        index=False
    )

# Combine all results
df_all = pd.concat(results, ignore_index=True)
df_all.to_csv("data/processed/all_documents_with_queries.csv", index=False)
```

## Example Output

For a document about "Device Trust 인증 가이드":

**Generated Queries:**
1. Device Trust 정책이 시행된 이후, 인증된 단말에서만 접근할 수 있는 애플리케이션의 중요도 기준은 무엇인가요?
2. Okta Verify 앱을 사용하여 장치를 등록하는 과정에서 문제가 발생하면 어디에 문의해야 하나요?
3. Windows에서 지문 인식을 설정하는 방법에 대해 자세히 설명해 주실 수 있나요?

## Cost Estimation

Using `gpt-4o-mini`:
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens

For 89 documents with average 1000 tokens each:
- Estimated cost: ~$0.10-0.50 total

## Troubleshooting

### Error: "OpenAI API key is required"
```bash
export OPENAI_API_KEY="your-key-here"
```

### Error: "DataFrame must contain 'page_content' column"
Ensure your CSV has a `page_content` column with the document text.

### Slow processing
- Reduce `num_queries` (e.g., from 3 to 2)
- Process in smaller batches
- Use `gpt-4o-mini` instead of `gpt-4o`

## Next Steps

After generating queries, you can use them to:
1. Train a Q&A retrieval system
2. Fine-tune a chatbot model
3. Create evaluation datasets
4. Build a test suite for your HR chatbot

For more information on the document classifier, see the main project README.