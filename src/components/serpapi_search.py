import logging
from typing import List, Dict, Any
from serpapi.google_search import GoogleSearch  # Fixed import

logger = logging.getLogger("ImageSearchPipeline")

class SerpApiImageSearch:
    """Class for searching images using SerpAPI"""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def search_images(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for images using SerpAPI, optimized for OpenAI Vision API compatibility
        
        OpenAI Vision API supports:
        - PNG (.png)
        - JPEG (.jpeg and .jpg)
        - WEBP (.webp)
        - Non-animated GIF (.gif)
        """
        try:
            # Modify query to prioritize supported formats
            optimized_query = f"{query} filetype:jpg OR filetype:png OR filetype:webp"
            
            search_params = {
                "engine": "google_images",
                "q": optimized_query,
                "api_key": self.api_key,
                # Request more than we need to allow for filtering
                "num": min(num_results * 2, 20),  # Cap at 20 to limit API costs
                # Add image size parameter to avoid very small images
                "image_size": "large",
                # Prioritize photos over illustrations, clipart, etc.
                "image_type": "photo"
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            images = []
            if "images_results" in results:
                # Pre-filter for compatible formats
                compatible_images = []
                for item in results["images_results"]:
                    url = item.get("original", "")
                    # Check if URL has compatible extension
                    if any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                        compatible_images.append(item)
                    
                    # Break early if we have enough images
                    if len(compatible_images) >= num_results:
                        break
                
                # Process the compatible images
                for item in compatible_images[:num_results]:
                    image_data = {
                        "title": item.get("title", ""),
                        "url": item.get("original", ""),
                        "source_page": item.get("source", ""),
                        "thumbnail": item.get("thumbnail", ""),
                        "width": item.get("original_width", 0),
                        "height": item.get("original_height", 0)
                    }
                    images.append(image_data)
            
            return images
            
        except Exception as e:
            logger.error(f"Error searching for images via SerpAPI: {e}")
            return []