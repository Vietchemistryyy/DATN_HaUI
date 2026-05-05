"""
URL Scraper using Trafilatura with BeautifulSoup Fallback
"""
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class URLScraper:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False

    def scrape(self, url: str) -> Dict:
        if not self._is_valid_url(url):
            return {"success": False, "text": None, "error": "Invalid URL format"}
        
        logger.info(f"Scraping URL: {url}")
        
        # Strategy 1: Trafilatura (Smart extraction)
        try:
            downloaded = trafilatura.fetch_url(url, timeout=self.timeout)
            if downloaded:
                text = trafilatura.extract(downloaded, deduplicate=True, include_comments=False)
                if text and len(text) >= 100:
                    text = self._clean_text(text)
                    return {"success": True, "text": text, "method": "trafilatura", "error": None}
        except Exception as e:
            logger.warning(f"Trafilatura failed: {e}")

        # Strategy 2: BeautifulSoup (Fallback)
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove scripts, styles, nav, headers, footers
            for elem in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
                elem.decompose()
                
            article_text = None
            
            # Try <article> tag
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs])
                
            # Try <main> or <div role="main">
            if not article_text or len(article_text) < 200:
                main_content = soup.find('main') or soup.find('div', {'role': 'main'})
                if main_content:
                    paragraphs = main_content.find_all('p')
                    text = ' '.join([p.get_text().strip() for p in paragraphs])
                    if len(text) > (len(article_text) if article_text else 0):
                        article_text = text
            
            # Fallback to all paragraphs
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all('p')
                paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                article_text = ' '.join(paragraphs)
                
            article_text = self._clean_text(article_text)
            
            if article_text and len(article_text) >= 100:
                return {"success": True, "text": article_text, "method": "beautifulsoup", "error": None}
            else:
                return {"success": False, "text": None, "error": "Could not extract sufficient text content."}
                
        except Exception as e:
            logger.error(f"BeautifulSoup fallback failed: {e}")
            error_msg = str(e)
            if "401 Client Error" in error_msg or "403 Client Error" in error_msg:
                friendly_err = "Access blocked by the website's anti-bot protection (HTTP 401/403). Please manually copy the article and use the 'Paste Text' feature."
                return {"success": False, "text": None, "error": friendly_err}
            return {"success": False, "text": None, "error": f"Failed to fetch content: {error_msg}"}

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def scrape_url(url: str, timeout: int = 15) -> Dict:
    scraper = URLScraper(timeout=timeout)
    return scraper.scrape(url)
