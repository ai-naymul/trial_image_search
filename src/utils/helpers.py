import os
from typing import List

def get_image_files(directory: str) -> List[str]:
    """Get list of image files in a directory"""
    if not os.path.exists(directory):
        return []
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file_path)
    
    return image_files

def is_valid_image(file_path: str) -> bool:
    """Check if a file is a valid image"""
    try:
        from PIL import Image
        Image.open(file_path).verify()
        return True
    except:
        return False