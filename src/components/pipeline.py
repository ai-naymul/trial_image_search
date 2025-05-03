import os
import logging
from typing import List
from dataclasses import dataclass

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from src.components.image_analyzer import ImageAnalyzer
from src.components.description_parser import DescriptionParser
from src.components.vector_store import VectorStoreManager

# Configure logging
logger = logging.getLogger("ImageSearchPipeline")

@dataclass
class SearchResult:
    """Class for storing search result information"""
    path: str
    filename: str
    match_score: float
    similarity: float

class ImageSearchPipeline:
    """Main pipeline class for finding images matching descriptions"""
    
    def __init__(self, openai_api_key: str, image_folder_path: str):
        self.openai_api_key = openai_api_key
        self.image_folder_path = image_folder_path
        
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
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize component classes
        self.image_analyzer = ImageAnalyzer(self.vision_llm)
        self.description_parser = DescriptionParser(self.text_llm)
        self.vector_store_manager = VectorStoreManager(self.embeddings, self.image_analyzer)
        
        # Initialize vector store
        logger.info(f"Initializing vector store from {image_folder_path}...")
        self.vector_store_manager.initialize_from_directory(image_folder_path)
        logger.info("Pipeline initialization complete.")
    
    def search(self, description: str, top_k: int = 5) -> List[SearchResult]:
        """Search for images matching the given description"""
        logger.info(f"Searching for images matching: {description}")
        
        try:
            # Step 1: Break down description into components
            logger.info("Breaking down description into components...")
            description_components = self.description_parser.break_down_description(description)
            
            # Step 2: Generate search queries
            logger.info("Generating search queries...")
            search_queries = self.description_parser.generate_search_queries(description_components)
            logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")
            
            # Step 3: Perform vector searches for each query
            logger.info("Performing vector searches...")
            all_results = {}
            
            for query in search_queries:
                logger.info(f"Searching for: {query}")
                results = self.vector_store_manager.similarity_search(query, k=top_k*2)
                
                for doc, score in results:
                    img_path = doc.metadata["source"]
                    # Track the best similarity score for each image
                    if img_path not in all_results or score > all_results[img_path]["similarity"]:
                        all_results[img_path] = {
                            "similarity": score,
                            "path": img_path
                        }
            
            # Get top candidates based on similarity
            candidates = sorted(all_results.values(), key=lambda x: x["similarity"], reverse=True)[:top_k*2]
            logger.info(f"Found {len(candidates)} initial candidates based on similarity")
            
            # Step 4: Evaluate candidates for precise matching
            logger.info("Evaluating candidates for precise matching...")
            final_results = []
            
            for candidate in candidates:
                img_path = candidate["path"]
                logger.info(f"Evaluating {os.path.basename(img_path)}...")
                
                # Evaluate match score
                match_score = self.image_analyzer.evaluate_match(img_path, description_components)
                
                # Create result object
                result = SearchResult(
                    path=img_path,
                    filename=os.path.basename(img_path),
                    match_score=match_score,
                    similarity=candidate["similarity"]
                )
                
                final_results.append(result)
            
            # Sort by match score
            final_results.sort(key=lambda x: x.match_score, reverse=True)
            
            # Return top-k results
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search pipeline: {e}")
            return []