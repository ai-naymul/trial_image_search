import os
import logging
from typing import List, Tuple

from PIL import Image
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("ImageSearchPipeline")

class VectorStoreManager:
    """Class for managing vector store operations"""
    
    def __init__(self, embeddings, image_analyzer):
        self.embeddings = embeddings
        self.image_analyzer = image_analyzer
        self.vector_store = None
    
    def initialize_from_directory(self, image_folder_path: str):
        """Initialize vector store from a directory of images"""
        try:
            if not os.path.exists(image_folder_path):
                logger.error(f"Image folder path does not exist: {image_folder_path}")
                raise FileNotFoundError(f"Image folder path does not exist: {image_folder_path}")
            
            # Get all image files
            image_files = []
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            
            for file in os.listdir(image_folder_path):
                file_path = os.path.join(image_folder_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_files.append(file_path)
            
            if not image_files:
                logger.warning(f"No image files found in {image_folder_path}")
                return
            
            # Generate descriptions for all images
            logger.info(f"Generating descriptions for {len(image_files)} images...")
            documents = []
            
            for i, img_path in enumerate(image_files, 1):
                logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(img_path)}")
                
                try:
                    # Check if image file is valid
                    Image.open(img_path).verify()
                    
                    # Generate description
                    description = self.image_analyzer.generate_description(img_path)
                    
                    # Create document
                    document = Document(
                        page_content=description,
                        metadata={"source": img_path, "filename": os.path.basename(img_path)}
                    )
                    
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
            
            # Create vector store
            logger.info("Creating vector store from image descriptions...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("Vector store initialization complete.")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform similarity search in the vector store"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []