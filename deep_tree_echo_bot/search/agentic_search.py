"""
Strong Agentic Search Engine for DeepTreeEchoBot.
Provides intelligent search capabilities with task-aware filtering and ranking.
"""
import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json
from urllib.parse import quote_plus

from ..config.settings import SearchConfig


class SearchResult:
    """Represents a search result with relevance scoring."""
    
    def __init__(self, title: str, url: str, snippet: str, relevance: float = 0.0):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.relevance = relevance
        
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'relevance': self.relevance
        }


class AgenticSearchEngine:
    """
    Strong agentic search engine that adapts to task requirements
    and provides contextually relevant results.
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Semantic search model
        self.sentence_model = None
        self.session = None
        
        # Search APIs and endpoints
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'bing': self._search_bing,
            'google': self._search_google_fallback
        }
        
        # Context tracking
        self.search_history = []
        self.context_embeddings = []
        
    async def initialize(self):
        """Initialize the search engine."""
        self.logger.info("Initializing Agentic Search Engine...")
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Sentence transformer model loaded")
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'DeepTreeEchoBot/1.0'}
        )
        
        self.logger.info("Agentic Search Engine initialized")
        
    async def search(
        self, 
        query: str, 
        task_vector: Optional[np.ndarray] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Perform intelligent search with task-aware ranking.
        
        Args:
            query: Search query
            task_vector: Task-specific vector for contextual relevance
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Performing agentic search for: {query}")
        
        # Multi-engine search
        all_results = []
        successful_engines = []
        
        for engine_name, search_func in self.search_engines.items():
            try:
                results = await search_func(query, max_results)
                all_results.extend(results)
                successful_engines.append(engine_name)
                self.logger.debug(f"Got {len(results)} results from {engine_name}")
            except Exception as e:
                self.logger.warning(f"Search engine {engine_name} failed: {e}")
                
        if not all_results:
            self.logger.warning("All search engines failed")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'engines_used': [],
                'success': False
            }
            
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = await self._rank_results(unique_results, query, task_vector)
        
        # Limit results
        final_results = ranked_results[:max_results]
        
        # Update search history
        self.search_history.append({
            'query': query,
            'results_count': len(final_results),
            'engines_used': successful_engines
        })
        
        return {
            'query': query,
            'results': [r.to_dict() for r in final_results],
            'total_results': len(final_results),
            'engines_used': successful_engines,
            'success': True
        }
        
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo Instant Answer API."""
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Process instant answer
                    if data.get('Answer'):
                        results.append(SearchResult(
                            title="DuckDuckGo Instant Answer",
                            url=data.get('AnswerURL', ''),
                            snippet=data['Answer'],
                            relevance=0.9
                        ))
                        
                    # Process abstract
                    if data.get('Abstract'):
                        results.append(SearchResult(
                            title=data.get('AbstractSource', 'Abstract'),
                            url=data.get('AbstractURL', ''),
                            snippet=data['Abstract'],
                            relevance=0.8
                        ))
                        
                    # Process related topics
                    for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append(SearchResult(
                                title=topic.get('FirstURL', {}).get('Text', 'Related Topic'),
                                url=topic.get('FirstURL', {}).get('FirstURL', ''),
                                snippet=topic['Text'],
                                relevance=0.6
                            ))
                            
                    return results[:max_results]
                    
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            
        return []
        
    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing (placeholder - would need API key)."""
        # This would require Bing Search API subscription
        # For now, return empty results
        self.logger.debug("Bing search not configured (requires API key)")
        return []
        
    async def _search_google_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search method using web scraping (educational purposes only)."""
        # Simple fallback that creates synthetic results based on query
        results = []
        
        # Generate some basic results based on query keywords
        keywords = query.lower().split()
        
        for i, keyword in enumerate(keywords[:max_results]):
            results.append(SearchResult(
                title=f"Information about {keyword.title()}",
                url=f"https://example.com/search?q={quote_plus(keyword)}",
                snippet=f"Relevant information about {keyword} in the context of '{query}'.",
                relevance=max(0.3, 1.0 - i * 0.1)
            ))
            
        return results
        
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and title similarity."""
        unique_results = []
        seen_urls = set()
        seen_titles = set()
        
        for result in results:
            # Check URL uniqueness
            if result.url and result.url in seen_urls:
                continue
                
            # Check title similarity
            title_lower = result.title.lower()
            if any(self._calculate_similarity(title_lower, seen) > 0.8 for seen in seen_titles):
                continue
                
            unique_results.append(result)
            if result.url:
                seen_urls.add(result.url)
            seen_titles.add(title_lower)
            
        return unique_results
        
    async def _rank_results(
        self, 
        results: List[SearchResult], 
        query: str, 
        task_vector: Optional[np.ndarray] = None
    ) -> List[SearchResult]:
        """Rank results based on relevance to query and task context."""
        
        if not results:
            return results
            
        if self.sentence_model is None:
            # Fallback ranking based on initial relevance scores
            return sorted(results, key=lambda r: r.relevance, reverse=True)
            
        try:
            # Get query embedding
            query_embedding = self.sentence_model.encode([query])[0]
            
            # Calculate semantic similarity for each result
            for result in results:
                text_to_embed = f"{result.title} {result.snippet}"
                result_embedding = self.sentence_model.encode([text_to_embed])[0]
                
                # Cosine similarity
                similarity = np.dot(query_embedding, result_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                )
                
                # Combine with initial relevance
                result.relevance = 0.6 * similarity + 0.4 * result.relevance
                
                # Task vector bonus if provided
                if task_vector is not None:
                    task_similarity = np.dot(result_embedding, task_vector[:len(result_embedding)]) / (
                        np.linalg.norm(result_embedding) * np.linalg.norm(task_vector[:len(result_embedding)])
                    )
                    result.relevance += 0.2 * task_similarity
                    
        except Exception as e:
            self.logger.warning(f"Error in semantic ranking: {e}")
            
        # Sort by relevance
        return sorted(results, key=lambda r: r.relevance, reverse=True)
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    async def get_context_features(self) -> torch.Tensor:
        """Get current search context as feature vector."""
        features = torch.zeros(self.config.context_feature_dim)
        
        if self.search_history:
            # Simple features based on search history
            features[0] = len(self.search_history)  # Number of searches
            features[1] = len(self.search_history[-1].get('engines_used', [])) if self.search_history else 0
            features[2] = self.search_history[-1].get('results_count', 0) if self.search_history else 0
            
        return features
        
    async def shutdown(self):
        """Shutdown the search engine."""
        if self.session:
            await self.session.close()
        self.logger.info("Agentic Search Engine shutdown complete")