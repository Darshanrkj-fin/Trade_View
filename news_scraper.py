"""
news_scraper.py - Unified News Collection Module
Scrapes financial news from LiveMint and Google News for Indian stock market analysis.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common headers to avoid bot detection
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def fetch_with_retries(url, timeout=15, attempts=2):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Request attempt %s/%s failed for %s: %s", attempt, attempts, url, exc)
            time.sleep(0.4)
    raise last_error


def scrape_livemint_news():
    """
    Scrape financial news from LiveMint across multiple sections.
    Returns a list of dicts with headline, text, link, source.
    """
    sections = {
        'companies': 'https://www.livemint.com/companies/news',
        'commodities': 'https://www.livemint.com/market/commodities',
        'stock_market': 'https://www.livemint.com/market/stock-market-news',
        'ipo': 'https://www.livemint.com/market/ipo',
        'mark_to_market': 'https://www.livemint.com/market/mark-to-market',
    }

    articles = []

    for section_name, url in sections.items():
        try:
            logger.info(f"Scraping LiveMint: {section_name}...")
            response = fetch_with_retries(url, timeout=15, attempts=2)

            soup = BeautifulSoup(response.text, 'html.parser')
            story_divs = soup.find_all('div', {'class': 'listtostory clearfix'})

            for div in story_divs:
                try:
                    a_tag = div.find('a')
                    if not a_tag:
                        continue

                    href = a_tag.get('href', '')
                    href = href.replace('[', '').replace(']', '').replace('"', '').replace("'", '')

                    if not href.startswith('http'):
                        href = 'https://www.livemint.com/' + href.lstrip('/')

                    headline = a_tag.get_text(strip=True)
                    if not headline:
                        # Try to get headline from h2 or span inside
                        h2 = div.find('h2')
                        headline = h2.get_text(strip=True) if h2 else ''

                    if headline:
                        articles.append({
                            'headline': headline,
                            'text': '',  # Will be filled if we fetch full article
                            'link': href,
                            'source': f'LiveMint ({section_name})',
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.debug(f"Error parsing LiveMint article: {e}")
                    continue

            time.sleep(0.5)  # Rate limiting

        except requests.RequestException as e:
            logger.warning(f"Failed to scrape LiveMint {section_name}: {e}")
            continue

    # Optionally fetch article text for the first few articles
    for article in articles[:15]:  # Limit to 15 to avoid excessive requests
        try:
            resp = fetch_with_retries(article['link'], timeout=10, attempts=2)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text_parts = []
                for p in paragraphs[:3]:  # First 3 paragraphs
                    p_text = p.get_text(strip=True)
                    if p_text and 'Never miss a story!' not in p_text:
                        text_parts.append(p_text)
                article['text'] = ' '.join(text_parts)
            time.sleep(0.3)
        except Exception:
            pass

    logger.info(f"Collected {len(articles)} articles from LiveMint")
    return articles


def scrape_google_news(keyword="Indian stocks", days=7):
    """
    Scrape Google News for a given keyword.
    Returns a list of dicts with headline, link, source.
    """
    articles = []
    try:
        url = f"https://news.google.com/search?q={keyword}+when:{days}d&tbm=nws"
        logger.info(f"Scraping Google News for: {keyword}...")
        response = fetch_with_retries(url, timeout=15, attempts=2)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try multiple selectors as Google News changes frequently
            news_links = soup.find_all('a', class_='JtKRv')
            if not news_links:
                news_links = soup.find_all('a', {'class': re.compile(r'DY5T1d|JtKRv|VDXfz')})
            if not news_links:
                # Fallback: find article tags
                article_tags = soup.find_all('article')
                for art in article_tags:
                    a = art.find('a')
                    if a:
                        news_links.append(a)

            for link in news_links:
                headline = link.get_text(strip=True)
                href = link.get('href', '')
                if href.startswith('./'):
                    href = 'https://news.google.com/' + href[2:]

                if headline and len(headline) > 10:
                    articles.append({
                        'headline': headline,
                        'text': '',
                        'link': href,
                        'source': 'Google News',
                        'timestamp': datetime.now().isoformat()
                    })

            logger.info(f"Collected {len(articles)} articles from Google News for '{keyword}'")
        else:
            logger.warning(f"Google News returned status {response.status_code}")

    except Exception as e:
        logger.warning(f"Failed to scrape Google News: {e}")

    return articles


def scrape_google_news_for_stocks(stock_keywords=None):
    """
    Scrape Google News for multiple Indian stock related keywords.
    """
    if stock_keywords is None:
        stock_keywords = [
            "Indian stock market",
            "NSE BSE stocks",
            "Nifty 50 stocks",
            "Sensex stocks today",
            "Indian stocks buy sell",
            "stock market India news",
        ]

    all_articles = []
    seen_headlines = set()

    for keyword in stock_keywords:
        articles = scrape_google_news(keyword, days=3)
        for article in articles:
            # Deduplicate by headline
            headline_key = article['headline'].lower().strip()
            if headline_key not in seen_headlines:
                seen_headlines.add(headline_key)
                all_articles.append(article)
        time.sleep(1)  # Rate limiting between searches

    logger.info(f"Total unique articles from Google News: {len(all_articles)}")
    return all_articles


def scrape_all_news():
    """
    Main function: scrape all news sources and return a unified DataFrame.
    
    Returns:
        pd.DataFrame with columns: headline, text, link, source, timestamp
    """
    all_articles = []
    source_status = {"livemint": "unavailable", "google_news": "unavailable"}

    # 1. LiveMint
    try:
        livemint_articles = scrape_livemint_news()
        all_articles.extend(livemint_articles)
        if livemint_articles:
            source_status["livemint"] = "ok"
    except Exception as e:
        logger.error(f"LiveMint scraping failed: {e}")
        source_status["livemint"] = "failed"

    # 2. Google News
    try:
        google_articles = scrape_google_news_for_stocks()
        all_articles.extend(google_articles)
        if google_articles:
            source_status["google_news"] = "ok"
    except Exception as e:
        logger.error(f"Google News scraping failed: {e}")
        source_status["google_news"] = "failed"

    if not all_articles:
        logger.warning("No articles collected from any source!")
        return pd.DataFrame(columns=['headline', 'text', 'link', 'source', 'timestamp'])

    df = pd.DataFrame(all_articles)

    # Deduplicate by headline similarity
    df = df.drop_duplicates(subset='headline', keep='first').reset_index(drop=True)

    logger.info(f"Total scraped articles: {len(df)}")
    df.attrs["source_status"] = source_status
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("News Scraper - Testing Mode")
    print("=" * 60)
    df = scrape_all_news()
    print(f"\nCollected {len(df)} articles:")
    print(df[['headline', 'source']].to_string(max_rows=20))
