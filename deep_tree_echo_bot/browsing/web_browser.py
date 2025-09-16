"""
Basic Web Browsing Capabilities for DeepTreeEchoBot.
Provides efficient CPU-based web content retrieval and processing.
"""
import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Any
import numpy as np
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime

from ..config.settings import BrowsingConfig


class WebPage:
    """Represents a web page with extracted content."""
    
    def __init__(self, url: str, title: str, content: str, links: List[str], metadata: Dict):
        self.url = url
        self.title = title
        self.content = content
        self.links = links
        self.metadata = metadata
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content[:1000] + '...' if len(self.content) > 1000 else self.content,
            'links_count': len(self.links),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class WebBrowser:
    """
    Basic web browsing capabilities optimized for CPU performance.
    Focuses on efficient content extraction and text processing.
    """
    
    def __init__(self, config: BrowsingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # HTTP session
        self.session = None
        
        # Content cache
        self.page_cache = {}
        self.browsing_history = []
        
        # Content filters
        self.allowed_content_types = {
            'text/html',
            'text/plain',
            'application/xhtml+xml'
        }
        
    async def initialize(self):
        """Initialize the web browser."""
        self.logger.info("Initializing Web Browser...")
        
        # Configure HTTP session with reasonable timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=10,
            sock_read=30
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                'User-Agent': 'DeepTreeEchoBot/1.0 (Educational Purpose)',
                'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive'
            }
        )
        
        self.logger.info("Web Browser initialized")
        
    async def browse(
        self, 
        url: str, 
        task_vector: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Browse a web URL and extract relevant content.
        
        Args:
            url: URL to browse
            task_vector: Task-specific vector for content filtering
            
        Returns:
            Dictionary with browsing results and extracted content
        """
        self.logger.info(f"Browsing URL: {url}")
        
        # Check cache first
        if url in self.page_cache:
            cached_page = self.page_cache[url]
            self.logger.debug(f"Using cached content for {url}")
            return {
                'success': True,
                'url': url,
                'content': cached_page.content,
                'title': cached_page.title,
                'links': cached_page.links,
                'metadata': cached_page.metadata,
                'cached': True
            }
            
        try:
            # Fetch the page
            page = await self._fetch_page(url)
            
            if page:
                # Cache the page
                self.page_cache[url] = page
                
                # Update browsing history
                self.browsing_history.append({
                    'url': url,
                    'title': page.title,
                    'timestamp': page.timestamp,
                    'content_length': len(page.content)
                })
                
                # Filter content based on task vector if provided
                filtered_content = await self._filter_content(page.content, task_vector)
                
                return {
                    'success': True,
                    'url': url,
                    'content': filtered_content,
                    'title': page.title,
                    'links': page.links[:self.config.max_links_per_page],
                    'metadata': page.metadata,
                    'cached': False
                }
            else:
                return {
                    'success': False,
                    'url': url,
                    'error': 'Failed to fetch page content',
                    'content': '',
                    'title': '',
                    'links': [],
                    'metadata': {}
                }
                
        except Exception as e:
            self.logger.error(f"Error browsing {url}: {e}")
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'content': '',
                'title': '',
                'links': [],
                'metadata': {}
            }
            
    async def _fetch_page(self, url: str) -> Optional[WebPage]:
        """Fetch and parse a web page."""
        try:
            async with self.session.get(url) as response:
                # Check response status
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
                    
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in self.allowed_content_types):
                    self.logger.warning(f"Unsupported content type {content_type} for {url}")
                    return None
                    
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.config.max_page_size:
                    self.logger.warning(f"Page too large ({content_length} bytes) for {url}")
                    return None
                    
                # Read content
                html_content = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract page information
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                links = self._extract_links(soup, url)
                metadata = self._extract_metadata(soup, response)
                
                return WebPage(url, title, content, links, metadata)
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {url}")
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            
        return None
        
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
            
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        return "Untitled Page"
        
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main'))
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
                
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Limit content length
        if len(text) > self.config.max_content_length:
            text = text[:self.config.max_content_length] + '...'
            
        return text
        
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from the page."""
        links = []
        
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Filter out unwanted links
            if self._is_valid_link(absolute_url):
                links.append(absolute_url)
                
            if len(links) >= self.config.max_links_per_page:
                break
                
        return links
        
    def _is_valid_link(self, url: str) -> bool:
        """Check if a link is valid and useful."""
        try:
            parsed = urlparse(url)
            
            # Must have a scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Only HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
                
            # Skip common unwanted extensions
            unwanted_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                                 '.zip', '.rar', '.tar', '.gz', '.mp3', '.mp4', '.avi', '.jpg', 
                                 '.jpeg', '.png', '.gif', '.svg'}
            
            if any(url.lower().endswith(ext) for ext in unwanted_extensions):
                return False
                
            # Skip javascript and mailto links
            if url.startswith(('javascript:', 'mailto:')):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _extract_metadata(self, soup: BeautifulSoup, response) -> Dict:
        """Extract metadata from the page."""
        metadata = {
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(str(soup)),
        }
        
        # Extract meta tags
        for meta_tag in soup.find_all('meta'):
            name = meta_tag.get('name') or meta_tag.get('property')
            content = meta_tag.get('content')
            
            if name and content:
                metadata[f'meta_{name}'] = content
                
        # Extract headings
        headings = []
        for i in range(1, 4):  # h1, h2, h3
            for heading in soup.find_all(f'h{i}'):
                headings.append(heading.get_text().strip())
                
        metadata['headings'] = headings[:10]  # Limit to 10 headings
        
        return metadata
        
    async def _filter_content(self, content: str, task_vector: Optional[np.ndarray] = None) -> str:
        """Filter content based on task requirements."""
        if not task_vector:
            return content
            
        # Simple keyword-based filtering
        # In a more sophisticated implementation, this would use semantic similarity
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # For now, return first N sentences (simple filtering)
        max_sentences = min(10, len(sentences))
        filtered_sentences = sentences[:max_sentences]
        
        return '. '.join(s.strip() for s in filtered_sentences if s.strip()) + '.'
        
    async def get_browsing_history(self) -> List[Dict]:
        """Get browsing history."""
        return self.browsing_history.copy()
        
    async def clear_cache(self):
        """Clear the page cache."""
        self.page_cache.clear()
        self.logger.info("Page cache cleared")
        
    async def close(self):
        """Close the web browser and cleanup resources."""
        if self.session:
            await self.session.close()
            
        self.page_cache.clear()
        self.logger.info("Web Browser closed")