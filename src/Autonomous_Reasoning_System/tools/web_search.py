
import logging
from playwright.sync_api import sync_playwright
import urllib.parse
import time

logger = logging.getLogger(__name__)

def perform_google_search(query: str, **kwargs) -> str:
    """
    Performs a Google search using Playwright and returns the top results.
    """
    logger.info(f"Performing Google Search for: {query}")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()

            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}"

            page.goto(url, timeout=30000)

            # Wait for results to load
            try:
                page.wait_for_selector("#search", timeout=5000)
            except:
                logger.warning("Google search selector timeout")

            # Extract text from search results
            # We target standard result blocks
            results = []

            # Get main snippet
            # Using evaluate to get text content safer
            js_extract = """
            () => {
                const items = Array.from(document.querySelectorAll('.g'));
                return items.map(item => {
                    const title = item.querySelector('h3')?.innerText || '';
                    const snippet = item.querySelector('.VwiC3b')?.innerText || item.querySelector('.IsZvec')?.innerText || '';
                    const link = item.querySelector('a')?.href || '';
                    if (title && snippet) {
                        return `Title: ${title}\\nLink: ${link}\\nSnippet: ${snippet}`;
                    }
                    return null;
                }).filter(i => i !== null).slice(0, 5);
            }
            """

            extracted_items = page.evaluate(js_extract)

            if extracted_items:
                results.extend(extracted_items)

            # Also check for "featured snippet" or "knowledge panel"
            # .kp-header or similar
            # .xpdopen (sometimes used for quick answers)

            js_quick_answer = """
            () => {
                const answer = document.querySelector('.Iz6qV')?.innerText || document.querySelector('.hgKElc')?.innerText;
                if (answer) return "Quick Answer: " + answer;
                return null;
            }
            """
            quick_answer = page.evaluate(js_quick_answer)
            if quick_answer:
                results.insert(0, quick_answer)

            browser.close()

            if not results:
                return "No search results found."

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"Google Search failed: {e}")
            return f"Error performing search: {e}"
