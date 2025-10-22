"""
Example script to generate HR queries from people intelligence documents.

This script demonstrates how to use the generate_queries function to create
training data for an HR Q&A chatbot.
"""

import os

import pandas as pd

from flex_ml.models import generate_queries
from flex_ml.utils.path import PROCESSED_DATA_PATH, RAW_DATA_PATH

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")


def main():
    """Generate queries for HR documents."""

    # Load the data
    input_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} documents")

    # Generate queries for a small sample first (for testing)
    sample_size = 5
    print(f"\nGenerating queries for {sample_size} sample documents...")
    df_sample = df.head(sample_size).copy()

    # Generate queries
    df_with_queries = generate_queries(
        df_sample,
        num_queries=3,  # Generate 3 queries per document
        temperature=0.7,  # Slightly creative
    )

    # Display results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS")
    print("=" * 80)

    for idx, row in df_with_queries.iterrows():
        print(f"\nDocument {idx + 1}:")
        print(f"Title: {row['title']}")
        print(f"Category L1: {row['categoryL1']}")
        print(f"Category L2: {row['categoryL2']}")
        print(f"Content preview: {row['page_content'][:200]}...")
        print(f"\nGenerated Queries:")
        for i, query in enumerate(row["query"], 1):
            print(f"  {i}. {query}")
        print("-" * 80)

    # Save results
    output_path = os.path.join(PROCESSED_DATA_PATH, "sample_documents_with_queries.csv")
    df_with_queries.to_csv(output_path, index=False)
    print(f"\nSample results saved to: {output_path}")

    # Optionally, process all documents (uncomment below)
    # print(f"\nProcessing all {len(df)} documents...")
    # df_all_with_queries = generate_queries(df, num_queries=3, temperature=0.7)
    # output_all_path = os.path.join(PROCESSED_DATA_PATH, "all_documents_with_queries.csv")
    # df_all_with_queries.to_csv(output_all_path, index=False)
    # print(f"All results saved to: {output_all_path}")


if __name__ == "__main__":
    main()
