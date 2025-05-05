# src/components/pipeline.py
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from PIL import Image
import json
import os
import time
import concurrent.futures
from functools import lru_cache

from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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
        
    # Class-level cache for API responses to avoid duplicate calls across searches
    _global_cache = {}
    
    @staticmethod
    @lru_cache(maxsize=100)
    def _cached_api_call(func_name, key, *args, **kwargs):
        """Generic caching wrapper for expensive API calls"""
        cache_key = f"{func_name}_{key}"
        if cache_key in ImageSearchPipeline._global_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return ImageSearchPipeline._global_cache[cache_key]
        
        start_time = time.time()
        result = args[0](*args[1:], **kwargs)  # args[0] is the function to call
        end_time = time.time()
        logger.debug(f"API call to {func_name} took {end_time - start_time:.2f} seconds")
        
        ImageSearchPipeline._global_cache[cache_key] = result
        return result
    
    def search(self, description: str, top_k: int = 5) -> List[SearchResult]:
        """
        Fast search for images matching the given description
        Optimized for speed (10-15 second response time)
        """
        start_time = time.time()
        logger.info(f"Starting fast search for: {description}")
        
        try:
            # Step 1: Break down description into components (cached)
            logger.info("Breaking down description...")
            description_components = self._cached_api_call(
                "break_down_description", 
                description,
                self.description_parser.break_down_description, 
                description
            )
            
            # Step 2: Create a simplified query for faster results
            main_elements = []
            # Get the most important elements
            for comp in description_components["elements"]:
                if comp.get("importance", 0) >= 6:  # Lower threshold to get more elements
                    main_elements.append(comp["element"])
            
            # Add main subjects if we have space
            if len(main_elements) < 2 and description_components.get("subjects"):
                main_elements.extend(description_components["subjects"][:2])
                
            # Create a simplified query
            simplified_query = " ".join(main_elements)
            if description_components.get("setting"):
                simplified_query += f" {description_components['setting']}"
                
            logger.info(f"Using simplified query: {simplified_query}")
            
            # Step 3: Perform parallel image search and initial filtering
            all_images = {}  # Dict to deduplicate images by URL
            
            # Perform the initial image search
            try:
                logger.info(f"Searching for images...")
                results = self.image_search.search_images(simplified_query, num_results=12)
                
                # Filter results to only include promising candidates (fast filtering only)
                for image_data in results:
                    image_url = image_data["url"]
                    # Only include if it has a valid format and reasonable size
                    if not self._is_likely_image_url(image_url):
                        continue
                    all_images[image_url] = image_data
                    
                    # Limit to top 8 images for speed
                    if len(all_images) >= 8:
                        break
                        
            except Exception as e:
                logger.error(f"Error in image search: {e}")
                
            # Early return if no images found
            if not all_images:
                logger.warning("No images found")
                return []
                
            logger.info(f"Found {len(all_images)} candidate images")
            
            # Step 4: Parallel processing of images for speed
            # Process images in parallel to save time
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Define a function to process a single image
                def process_image(url):
                    try:
                        # Fast check - can we download this image quickly?
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        }
                        response = requests.head(url, headers=headers, timeout=3)
                        
                        # Skip if not a valid image or too large
                        content_type = response.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/webp']):
                            return url, None
                        
                        # Check for presence of main elements
                        # We use a smaller set of elements for faster processing
                        check_elements = main_elements[:2] if main_elements else [description_components.get("subjects", [""])[0]]
                        
                        element_scores = self.image_analyzer.check_elements_presence(
                            url, check_elements, is_url=True
                        )
                        
                        if not element_scores:
                            return url, None
                            
                        # Calculate preliminary score 
                        present_elements = sum(1 for score in element_scores.values() if score >= 5)
                        preliminary_score = present_elements / max(1, len(check_elements))
                        
                        # Return results
                        return url, {
                            "element_scores": element_scores,
                            "preliminary_score": preliminary_score
                        }
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        return url, None
                
                # Submit all URLs for parallel processing
                future_to_url = {executor.submit(process_image, url): url for url in all_images.keys()}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        url, result = future.result()
                        if result:
                            all_images[url].update(result)
                        else:
                            # Remove this URL if processing failed
                            all_images.pop(url, None)
                    except Exception as e:
                        logger.error(f"Error getting result for {url}: {e}")
                        all_images.pop(url, None)
            
            # Check if we have any valid images left
            if not all_images:
                logger.warning("No valid images after processing")
                return []
                
            # Sort by preliminary score
            candidates = sorted(
                all_images.values(), 
                key=lambda x: x.get("preliminary_score", 0),
                reverse=True
            )[:min(top_k, len(all_images))]
            
            # Step 5: Fast evaluation
            logger.info("Fast evaluation of top candidates...")
            final_results = []
            
            # Process in parallel for speed
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                def evaluate_image(img_data):
                    try:
                        url = img_data["url"]
                        # Since we're optimizing for speed, skip full evaluation
                        # and use preliminary score directly
                        match_score = img_data.get("preliminary_score", 0.5)
                        
                        # For top few images, do a more detailed check if time allows
                        if len(final_results) < 3 and time.time() - start_time < 12:
                            try:
                                detailed_score = self.image_analyzer.evaluate_match(
                                    url, description_components, is_url=True
                                )
                                # Blend with preliminary score
                                match_score = 0.6 * detailed_score + 0.4 * match_score
                            except Exception as eval_err:
                                logger.warning(f"Detailed evaluation failed: {eval_err}")
                                # Continue with preliminary score only
                        
                        return SearchResult(
                            title=img_data.get("title", "Untitled Image"),
                            url=url,
                            source_page=img_data.get("source_page", ""),
                            thumbnail=img_data.get("thumbnail", ""),
                            match_score=match_score,
                            width=img_data.get("width", 0),
                            height=img_data.get("height", 0),
                            element_scores=img_data.get("element_scores", {})
                        )
                    except Exception as e:
                        logger.error(f"Error evaluating {url}: {e}")
                        return None
                
                # Submit all candidates for parallel evaluation
                futures = [executor.submit(evaluate_image, img_data) for img_data in candidates]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            final_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in evaluation: {e}")
            
            # Sort by match score
            final_results.sort(key=lambda x: x.match_score, reverse=True)
            
            # Try to download images for top results if time permits
            elapsed_time = time.time() - start_time
            if elapsed_time < 12:  # Only download if we have time
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    def download_top_image(result, index):
                        try:
                            local_path = self.download_image(result.url)
                            return index, local_path
                        except Exception as e:
                            logger.error(f"Error downloading image: {e}")
                            return index, None
                    
                    # Only try to download top few images
                    futures = [executor.submit(download_top_image, result, i) 
                              for i, result in enumerate(final_results[:min(3, len(final_results))])]
                    
                    # Update results with local paths
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            index, local_path = future.result()
                            if local_path and index < len(final_results):
                                final_results[index].local_path = local_path
                        except Exception as e:
                            logger.error(f"Error handling download result: {e}")
            
            # Final timing info
            total_time = time.time() - start_time
            logger.info(f"Search completed in {total_time:.2f} seconds")
            
            # Return top-k results
            return final_results[:top_k]
        except Exception as e:
            logger.error(f"Error in search pipeline: {e}")
            elapsed_time = time.time() - start_time
            logger.info(f"Search failed after {elapsed_time:.2f} seconds")
            return []

    def _batch_items(self, items, batch_size=5):
        """Utility function to batch items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]