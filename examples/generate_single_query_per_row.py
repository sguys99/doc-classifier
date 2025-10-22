"""
Example script to generate a single query per document row.

This script generates exactly one query per document and stores it as a string
(not a list) in the 'query' column.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from flex_ml.models import QueryGenerator
from flex_ml.utils.path import PROCESSED_DATA_PATH, RAW_DATA_PATH

load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")


def generate_single_query_per_row(
    df: pd.DataFrame,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Generate a single query for each document row.

    Args:
        df: DataFrame with 'page_content' column (and optionally 'title' column).
        temperature: OpenAI temperature parameter. Default is 0.7.

    Returns:
        DataFrame with added 'query' column containing a single query string.
    """
    # Initialize generator with num_queries=1
    generator = QueryGenerator(num_queries=1)

    queries = []
    for idx, row in df.iterrows():
        print(f"Generating query for document {idx + 1}/{len(df)}...")

        # Generate queries (will be a list with 1 item)
        query_list = generator.generate(
            page_content=row["page_content"],
            title=row.get("title", ""),
            temperature=temperature,
        )

        # Extract the single query from the list
        single_query = query_list[0]
        queries.append(single_query)

    # Add query column to original DataFrame
    result_df = df.copy()
    result_df["query"] = queries

    return result_df


def main():
    """Generate a single query per HR document."""

    # Load the data
    input_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} documents")

    # Generate queries for a small sample first (for testing)
    sample_size = 100
    print(f"\nGenerating single query for {sample_size} sample documents...")
    df_sample = df.head(sample_size).copy()

    # Generate single query per row
    df_with_queries = generate_single_query_per_row(
        df_sample,
        temperature=0.7,
    )

    # Display results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (1 query per row)")
    print("=" * 80)

    for idx, row in df_with_queries.iterrows():
        print(f"\nDocument {idx + 1}:")
        print(f"Title: {row['title']}")
        print(f"Category L1: {row['categoryL1']}")
        print(f"Category L2: {row['categoryL2']}")
        print(f"Content preview: {row['page_content'][:200]}...")
        print(f"\nGenerated Query:")
        print(f"  {row['query']}")
        print("-" * 80)

    # Save results
    output_path = os.path.join(PROCESSED_DATA_PATH, "sample_single_query_per_row.csv")
    df_with_queries.to_csv(output_path, index=False)
    print(f"\nSample results saved to: {output_path}")

    # Verify data structure
    print("\n" + "=" * 80)
    print("DATA STRUCTURE")
    print("=" * 80)
    print(f"Type of query column: {type(df_with_queries['query'].iloc[0])}")
    print(f"Sample query value: '{df_with_queries['query'].iloc[0]}'")

    # Optionally, process all documents (uncomment below)
    # print(f"\nProcessing all {len(df)} documents...")
    # df_all_with_queries = generate_single_query_per_row(df, temperature=0.7)
    # output_all_path = os.path.join(PROCESSED_DATA_PATH, "all_single_query_per_row.csv")
    # df_all_with_queries.to_csv(output_all_path, index=False)
    # print(f"All results saved to: {output_all_path}")


if __name__ == "__main__":
    main()
