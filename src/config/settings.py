import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "images_directory": os.getenv("IMAGES_DIRECTORY", "./images"),
    "text_model": "gpt-4",
    "vision_model": "gpt-4o",
    "max_tokens": 1024,
    "temperature": 0,
    "top_k_results": 5,
    "log_level": "INFO"
}

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return DEFAULT_CONFIG

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration settings"""
    if not config.get("openai_api_key"):
        return False
    
    if not os.path.exists(config.get("images_directory", "")):
        return False
    
    return True