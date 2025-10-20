"""Test script to debug Notion page fetching."""

import html2text
import requests
from bs4 import BeautifulSoup

# Test with the first URL from the CSV
test_url = "https://www.notion.so/flexnotion/64092ce439474ec982f6cf88051f40c2?pvs=25"

print(f"Testing URL: {test_url}\n")

try:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(test_url, headers=headers, timeout=10)
    response.raise_for_status()

    print(f"Status Code: {response.status_code}")
    print(f"Content Type: {response.headers.get('content-type')}")
    print(f"Response Length: {len(response.content)} bytes\n")

    # Save raw HTML for inspection
    with open("test_response.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Raw HTML saved to test_response.html\n")

    # Parse HTML
    soup = BeautifulSoup(response.content, "html.parser")

    # Check for Notion-specific content
    notion_app = soup.find("div", {"id": "notion-app"})
    print(f"Found #notion-app div: {notion_app is not None}")

    # Get all text
    print("\n--- Plain text extraction ---")
    text = soup.get_text(separator="\n", strip=True)
    print(f"Text length: {len(text)}")
    print(f"First 500 chars:\n{text[:500]}\n")

    # Try markdown conversion
    print("\n--- Markdown conversion ---")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    markdown = h.handle(str(soup))
    print(f"Markdown length: {len(markdown)}")
    print(f"First 500 chars:\n{markdown[:500]}\n")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")