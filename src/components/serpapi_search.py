from logging import getLogger
import os
from typing import List, Dict, Any, Optional
import functools
import time

from dotenv import load_dotenv
from serpapi import GoogleSearch

logger = getLogger(__name__)

class SerpApiImageSearch:
    """SerpAPI based image search"""
    
    # Class level cache to persist across instances
    _search_cache = {}
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize SerpAPI with key"""
        load_dotenv()  # Load environment variables
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SerpAPI key is required but not provided")
            
    @functools.lru_cache(maxsize=50)
    def search_images(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for images using SerpAPI's Google Images search with optimizations for speed.
        
        Args:
            query: Search query
            num_results: Maximum number of results to return
            
        Returns:
            List of image data dictionaries with urls and metadata
        """
        # Check cache first
        cache_key = f"{query}_{num_results}"
        if cache_key in self._search_cache:
            logger.info(f"Cache hit for query: {query}")
            return self._search_cache[cache_key]
            
        start_time = time.time()
        logger.info(f"Searching for images with query: {query}")
        
        # Optimize search parameters for better compatibility with OpenAI requirements
        params = {
            "engine": "google_images",
            "q": query,
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            # Filter for high-quality, non-watermarked images:
            "tbs": "ic:color,itp:photo", # color images that are photos
            "api_key": self.api_key,
            # Request more results to ensure we have enough after filtering
            "num": min(num_results * 2, 100)  # Get extra to allow for filtering
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            raw_results = results.get("images_results", [])
            
            # Filter results for compatibility with OpenAI's Vision API
            filtered_results = []
            
            # Fast pre-filtering for obvious non-matches
            for image in raw_results:
                # Skip very small images (likely thumbnails with poor quality)
                if image.get("original_width", 0) < 300 or image.get("original_height", 0) < 300:
                    continue
                    
                # Skip images with problematic domains or stock photo sites (often watermarked)
                image_url = image.get("original", image.get("thumbnail", ""))
                if any(domain in image_url.lower() for domain in ["shutterstock", "gettyimages", "alamy", "123rf"]):
                    continue
                
                # Only include images with supported formats
                if not any(ext in image_url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                    # If no extension in URL, check content type if available
                    if not image.get("source", "").endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                        continue
                
                # Prepare clean result with all needed metadata
                result = {
                    "url": image.get("original", ""),
                    "thumbnail": image.get("thumbnail", ""),
                    "title": image.get("title", ""),
                    "source_page": image.get("source", ""),
                    "width": image.get("original_width", 0),
                    "height": image.get("original_height", 0)
                }
                
                filtered_results.append(result)
                
                # Stop once we have enough results
                if len(filtered_results) >= num_results:
                    break
            
            # Cache the filtered results
            self._search_cache[cache_key] = filtered_results
            
            elapsed_time = time.time() - start_time
            logger.info(f"SerpAPI search completed in {elapsed_time:.2f} seconds, found {len(filtered_results)} images")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in SerpAPI search: {e}")
            return []