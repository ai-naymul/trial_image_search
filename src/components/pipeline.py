# src/components/pipeline.py
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from src.components.image_analyzer import ImageAnalyzer
from src.components.description_parser import DescriptionParser
from src.components.serpapi_search import SerpApiImageSearch

# Configure logging
logger = logging.getLogger("ImageSearchPipeline")

@dataclass
class SearchResult:
    """Class for storing search result information"""
    title: str
    url: str
    source_page: str
    thumbnail: str
    match_score: float
    local_path: Optional[str] = None
    width: int = 0
    height: int = 0
    element_scores: Dict[str, float] = field(default_factory=dict)


class ImageSearchPipeline:
    """Main pipeline class for finding images matching descriptions"""
    
    def __init__(self, openai_api_key: str, serpapi_key: str):
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # Initialize language models
        self.text_llm = ChatOpenAI(
            model_name="gpt-4", 
            temperature=0, 
            openai_api_key=openai_api_key
        )
        
        self.vision_llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            max_tokens=1024,
            openai_api_key=openai_api_key
        )
        
        # Initialize component classes
        self.image_analyzer = ImageAnalyzer(self.vision_llm)
        self.description_parser = DescriptionParser(self.text_llm)
        self.image_search = SerpApiImageSearch(serpapi_key)
        
        logger.info("Pipeline initialization complete.")
    
    def download_image(self, image_url: str) -> Optional[str]:
        """Download image from URL and save to a temporary file"""
        try:
            # Add User-Agent header to avoid some 403 errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check content type to ensure it's an image
            content_type = response.headers.get('content-type', '')
            if not any(img_type in content_type.lower() for img_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']):
                logger.warning(f"URL doesn't return an image content type: {content_type}")
                return None
            
            # Determine file extension from content type
            ext = '.jpg'  # Default extension
            if 'png' in content_type.lower():
                ext = '.png'
            elif 'gif' in content_type.lower():
                ext = '.gif'
            elif 'webp' in content_type.lower():
                ext = '.webp'
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_file.write(response.content)
            temp_file.close()
            
            # Verify it's a valid image
            try:
                img = Image.open(temp_file.name)
                img.verify()
                return temp_file.name
            except Exception as e:
                logger.error(f"Downloaded file is not a valid image: {e}")
                os.unlink(temp_file.name)  # Delete the invalid file
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"General error downloading image from {image_url}: {e}")
            return None
    
    def _is_likely_image_url(self, url: str) -> bool:
        """
        Check if a URL is likely to point to an image compatible with OpenAI's Vision API.
        
        OpenAI Vision API supports:
        - PNG (.png)
        - JPEG (.jpeg and .jpg)
        - WEBP (.webp)
        - Non-animated GIF (.gif)
        """
        # Check for supported image extensions
        supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        if any(url.lower().endswith(ext) for ext in supported_extensions):
            return True
            
        # Check for URL parameters indicating image format
        format_params = [
            'format=jpg', 'format=jpeg', 'format=png', 
            'format=webp', 'format=gif', 'type=image'
        ]
        if any(param in url.lower() for param in format_params):
            return True
            
        # Check for common image path patterns in URLs
        # These are less reliable but help identify likely image URLs
        path_indicators = ['images/', 'photos/', '/img/', '/image/', 'media/']
        if any(indicator in url.lower() for indicator in path_indicators):
            return True
            
        # Check for CDNs or services typically used for images
        cdn_patterns = ['cloudfront.net', 'cloudinary.com', 'imgix.net', 
                      'cdn.', 'images.', 'img.', 'photos.']
        if any(cdn in url.lower() for cdn in cdn_patterns):
            return True
            
        # Skip URLs explicitly for non-supported formats
        non_supported = ['.svg', '.tiff', '.bmp', '.ico', '.pdf', '.eps']
        if any(url.lower().endswith(ext) for ext in non_supported):
            return False
            
        return False
        
    # Cache for API responses to avoid duplicate calls
    _api_response_cache = {}
    
    def search(self, description: str, top_k: int = 5) -> List[SearchResult]:
        """Search for images matching the given description"""
        
        # Clear cache for new search
        self._api_response_cache = {}
        
        logger.info(f"Searching for images matching: {description}")
        
        try:
            # Step 1: Break down description into components
            logger.info("Breaking down description into components...")
            description_components = self.description_parser.break_down_description(description)
            
            # Step 2: Generate structured search queries
            logger.info("Generating search queries...")
            search_queries = self.description_parser.generate_search_queries(description_components)
            
            # Step 3: Perform image searches for each query
            logger.info("Performing image searches...")
            all_images = {}  # Dict to deduplicate images by URL
            
            # Limit the number of queries to reduce API calls
            # Use only the highest weighted queries
            search_queries.sort(key=lambda x: x["weight"], reverse=True)
            top_queries = search_queries[:min(3, len(search_queries))]
            
            for query_data in top_queries:
                query = query_data["query"]
                query_weight = query_data["weight"]
                query_elements = query_data["elements"]
                
                logger.info(f"Searching for images with query: {query}")
                try:
                    # Reduced number of results to just what we need (5-8 per query)
                    results = self.image_search.search_images(query, num_results=6)
                    
                    for image_data in results:
                        image_url = image_data["url"]
                        # Skip if we've already seen this image
                        if image_url in all_images:
                            continue
                            
                        # Store image with its query context
                        image_data["query"] = query
                        image_data["query_weight"] = query_weight
                        image_data["query_elements"] = query_elements
                        all_images[image_url] = image_data
                except Exception as e:
                    logger.error(f"Error searching with query '{query}': {e}")
                    continue
            
            logger.info(f"Found {len(all_images)} unique images from searches")
            
            # Early return if no images found
            if not all_images:
                return []
            
            # Step 4: Pre-filter by checking for essential elements
            logger.info("Pre-filtering images for essential elements...")
            essential_elements = []
            for comp in description_components["elements"]:
                if comp.get("importance", 0) >= 7:  # Consider high importance elements as essential
                    essential_elements.append(comp["element"])
            
            # Limit to essential elements if we have too many
            if len(essential_elements) > 3:
                essential_elements = essential_elements[:3]
            
            # If we don't have enough, add more from the subjects
            if len(essential_elements) < 2 and description_components.get("subjects"):
                for subj in description_components["subjects"]:
                    if subj not in essential_elements:
                        essential_elements.append(subj)
                        if len(essential_elements) >= 2:
                            break
            
            # Process only a limited number of images to check for essential elements
            # Sort the URLs by relevance based on the search query
            sorted_urls = sorted(
                all_images.keys(),
                key=lambda url: all_images[url]["query_weight"], 
                reverse=True
            )
            
            # Take only the top images to process (2-3 per query)
            top_urls = sorted_urls[:min(10, len(sorted_urls))]
            
            skipped_urls = []
            for url in top_urls:
                try:
                    # Check if we've already processed this URL (use cache)
                    cache_key = f"element_presence_{url}_{','.join(essential_elements)}"
                    if cache_key in self._api_response_cache:
                        element_scores = self._api_response_cache[cache_key]
                    else:
                        # Check for presence of essential elements
                        element_scores = self.image_analyzer.check_elements_presence(
                            url, essential_elements, is_url=True
                        )
                        # Cache the result
                        self._api_response_cache[cache_key] = element_scores
                    
                    # If we got back empty scores, skip this image
                    if not element_scores:
                        logger.info(f"No element scores returned for {url}, skipping")
                        skipped_urls.append(url)
                        continue
                        
                    # Store element scores
                    all_images[url]["element_scores"] = element_scores
                    
                    # Calculate preliminary score based on element presence
                    present_elements = sum(1 for score in element_scores.values() if score >= 5)
                    all_images[url]["preliminary_score"] = present_elements / max(1, len(essential_elements))
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    skipped_urls.append(url)
            
            # Remove skipped URLs from all_images
            for url in skipped_urls:
                if url in all_images:
                    del all_images[url]
                    
            # If we have too few images after filtering, we'll proceed with what we have
            if not all_images:
                logger.warning("No valid images remained after filtering")
                return []
            
            # Sort by preliminary score and select a very limited set of top candidates for detailed evaluation
            # This significantly reduces API calls
            candidates = sorted(
                all_images.values(), 
                key=lambda x: x.get("preliminary_score", 0),
                reverse=True
            )[:min(top_k+1, len(all_images))]  # Just top_k+1 to ensure we have enough after filtering
            
            # Step 5: Full evaluation for top candidates
            logger.info("Performing detailed evaluation on top candidates...")
            final_results = []
            
            for image_data in candidates:
                try:
                    url = image_data["url"]
                    
                    # Check cache first
                    cache_key = f"match_score_{url}_{description}"
                    if cache_key in self._api_response_cache:
                        match_score = self._api_response_cache[cache_key]
                    else:
                        # Perform full evaluation
                        match_score = self.image_analyzer.evaluate_match(
                            url,
                            description_components,
                            is_url=True
                        )
                        # Cache the result
                        self._api_response_cache[cache_key] = match_score
                    
                    # Blend with preliminary score for a more balanced result
                    final_score = 0.7 * match_score + 0.3 * image_data.get("preliminary_score", 0)
                    
                    result = SearchResult(
                        title=image_data["title"] or "Untitled Image",
                        url=image_data["url"],
                        source_page=image_data.get("source_page", ""),
                        thumbnail=image_data.get("thumbnail", ""),
                        match_score=final_score,
                        width=image_data.get("width", 0),
                        height=image_data.get("height", 0),
                        element_scores=image_data.get("element_scores", {})
                    )
                    
                    final_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating image {image_data.get('url', 'unknown')}: {e}")
                    continue
            
            # Sort by match score
            final_results.sort(key=lambda x: x.match_score, reverse=True)
            
            # Try to download images for top results
            for i, result in enumerate(final_results[:top_k]):
                try:
                    local_path = self.download_image(result.url)
                    if local_path:
                        final_results[i].local_path = local_path
                except Exception as e:
                    logger.error(f"Error downloading image {result.url}: {e}")
            
            # Return top-k results
            return final_results[:top_k]
        except Exception as e:
            logger.error(f"Error in search pipeline: {e}")
            return []

    def _batch_items(self, items, batch_size=5):
        """Utility function to batch items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]