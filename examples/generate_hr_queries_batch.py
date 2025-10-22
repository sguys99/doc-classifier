"""
Batch processing script to generate HR queries for all documents.

This script processes the entire people_intelligence_documents.csv file
in batches to avoid memory issues and provide progress tracking.
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from flex_ml.models import generate_queries
from flex_ml.utils.path import PROCESSED_DATA_PATH, RAW_DATA_PATH

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")


def main():
    """Generate queries for all HR documents in batches."""

    parser = argparse.ArgumentParser(description="Generate HR queries from documents")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of queries to generate per document (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from document index (useful for resuming, default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of documents to process (default: all)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated with timestamp)",
    )

    args = parser.parse_args()

    # Load the data
    input_path = os.path.join(RAW_DATA_PATH, "people_intelligence_documents.csv")
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} documents")

    # Apply start and limit
    if args.start_from > 0:
        print(f"Starting from document index: {args.start_from}")
        df = df.iloc[args.start_from :].copy()

    if args.limit:
        print(f"Limiting to {args.limit} documents")
        df = df.head(args.limit)

    total_docs = len(df)
    print(f"\nProcessing {total_docs} documents...")
    print(f"Batch size: {args.batch_size}")
    print(f"Queries per document: {args.num_queries}")
    print(f"Temperature: {args.temperature}")

    # Process in batches
    results = []
    batch_size = args.batch_size
    num_batches = (total_docs + batch_size - 1) // batch_size

    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for i in range(0, total_docs, batch_size):
        batch_num = i // batch_size + 1
        batch = df.iloc[i : i + batch_size].copy()

        print(f"\nProcessing batch {batch_num}/{num_batches} (docs {i+1}-{i+len(batch)})...")

        try:
            batch_with_queries = generate_queries(
                batch,
                num_queries=args.num_queries,
                temperature=args.temperature,
            )
            results.append(batch_with_queries)

            # Save intermediate results
            intermediate_path = os.path.join(PROCESSED_DATA_PATH, f"batch_{batch_num:03d}_queries.csv")
            batch_with_queries.to_csv(intermediate_path, index=False)
            print(f"✓ Batch {batch_num} saved to: {intermediate_path}")

        except Exception as e:
            print(f"✗ Error processing batch {batch_num}: {e}")
            print(f"Stopping at document index: {i + args.start_from}")
            break

    # Combine all results
    if results:
        print("\n" + "=" * 80)
        print("Combining all batches...")
        df_all = pd.concat(results, ignore_index=True)

        # Generate output filename
        if args.output_name:
            output_filename = args.output_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"documents_with_queries_{timestamp}.csv"

        output_path = os.path.join(PROCESSED_DATA_PATH, output_filename)
        df_all.to_csv(output_path, index=False)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'=' * 80}")
        print("COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")
        print(f"Total documents processed: {len(df_all)}")
        print(f"Total queries generated: {len(df_all) * args.num_queries}")
        print(f"Duration: {duration}")
        print(f"Output saved to: {output_path}")

        # Display sample results
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULTS (first document)")
        print(f"{'=' * 80}")
        sample = df_all.iloc[0]
        print(f"Title: {sample['title']}")
        print(f"Category: {sample['categoryL1']} / {sample['categoryL2']}")
        print(f"\nGenerated Queries:")
        for idx, query in enumerate(sample["query"], 1):
            print(f"  {idx}. {query}")

    else:
        print("\n✗ No results generated")


if __name__ == "__main__":
    main()
