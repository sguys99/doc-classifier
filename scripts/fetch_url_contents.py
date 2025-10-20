"""
Script to fetch webpage content from URLs in the help desk CSV file.
Adds a 'contents' column with the extracted text from each URL.
Uses Playwright for JavaScript-rendered Notion pages.
"""

import time
from pathlib import Path

import html2text
import pandas as pd
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from flex_ml.utils.path import PROCESSED_DATA_PATH, RAW_DATA_PATH


def fetch_webpage_content(url: str, timeout: int = 30000) -> str:
    """
    Fetch and extract text content from a Notion webpage URL in markdown format.
    Uses Playwright to handle JavaScript-rendered content.

    Args:
        url: The Notion URL to fetch content from
        timeout: Page load timeout in milliseconds (default: 30000ms)

    Returns:
        Extracted markdown content or error message
    """
    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the URL and wait for content to load
            page.goto(url, wait_until="networkidle", timeout=timeout)

            # Wait for Notion content to be rendered
            # Notion uses div with class containing 'notion-page-content'
            page.wait_for_selector("div.notion-page-content", timeout=10000)

            # Get the rendered HTML
            html_content = page.content()

            browser.close()

        # Initialize html2text converter for markdown conversion
        h = html2text.HTML2Text()
        h.ignore_links = False  # Keep links in markdown format
        h.ignore_images = False  # Keep images in markdown format
        h.ignore_emphasis = False  # Keep bold/italic formatting
        h.body_width = 0  # Don't wrap text
        h.skip_internal_links = True  # Skip anchor links
        h.ignore_tables = False  # Convert tables to markdown

        # Convert HTML to markdown
        markdown_text = h.handle(html_content)

        # Clean up extra whitespace while preserving markdown structure
        lines = markdown_text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.rstrip()
            cleaned_lines.append(stripped)

        # Remove excessive blank lines (more than 2 consecutive)
        result_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        markdown_text = "\n".join(result_lines).strip()

        return markdown_text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)} for {url}"


def main():
    """Main function to process the CSV file and fetch URL contents."""
    # Load the CSV file
    input_file = RAW_DATA_PATH / "help_desk_agent_poc.csv"
    output_file = PROCESSED_DATA_PATH / "help_desk_agent_poc_with_contents.csv"

    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    print(f"Found {len(df)} rows with URLs to process")

    # Create contents column
    contents = []

    # Fetch content for each URL with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching URLs"):
        url = row["url"]
        content = fetch_webpage_content(url)
        contents.append(content)

        # Rate limiting: wait 1 second between requests
        if idx < len(df) - 1:  # Don't wait after the last request
            time.sleep(1)

    # Add contents column to dataframe
    df["contents"] = contents

    # Save to processed data directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nProcessing complete!")
    print(f"Output saved to: {output_file}")

    # Show statistics
    error_count = sum(1 for c in contents if c.startswith("ERROR:"))
    success_count = len(contents) - error_count
    print(f"\nStatistics:")
    print(f"  Successfully fetched: {success_count}")
    print(f"  Errors: {error_count}")

    # Show sample of first successful fetch
    for idx, content in enumerate(contents):
        if not content.startswith("ERROR:"):
            print(f"\nSample content (row {idx}):")
            print(f"Title: {df.iloc[idx]['title']}")
            print(f"URL: {df.iloc[idx]['url']}")
            print(f"Content preview (first 200 chars):\n{content[:200]}...")
            break


if __name__ == "__main__":
    main()