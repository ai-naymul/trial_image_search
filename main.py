import os
import sys
import argparse
from dotenv import load_dotenv

# Add the current directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.components.pipeline import ImageSearchPipeline
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging()

def main():
    """Main entry point for the image search CLI"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Image Search Pipeline")
    parser.add_argument(
        "--description", "-d", 
        required=True,
        help="Detailed image description to search for"
    )
    parser.add_argument(
        "--images", "-i", 
        default=os.getenv("IMAGES_DIRECTORY", "./images"),
        help="Directory containing images to search"
    )
    parser.add_argument(
        "--api-key", "-k", 
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key"
    )
    parser.add_argument(
        "--results", "-r", 
        type=int, 
        default=3,
        help="Number of results to return"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.api_key:
        print("Error: OpenAI API key is required")
        return 1
    
    if not os.path.exists(args.images):
        print(f"Error: Image directory not found: {args.images}")
        return 1
    
    # Initialize the pipeline
    try:
        pipeline = ImageSearchPipeline(args.api_key, args.images)
        
        # Search for images
        results = pipeline.search(args.description, top_k=args.results)
        
        # Display results
        print(f"\nSearch Results for: {args.description}")
        print("="*50)
        
        if not results:
            print("No matching images found.")
            return 0
        
        for i, result in enumerate(results, 1):
            print(f"\nResult #{i}")
            print(f"Filename: {result.filename}")
            print(f"Path: {result.path}")
            print(f"Match Score: {result.match_score:.2f}")
            print(f"Similarity: {result.similarity:.2f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())