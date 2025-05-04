import base64
import json
import logging
import re
import tempfile
import requests
from typing import Dict, Any, Optional, List

from langchain.schema import HumanMessage, SystemMessage
from PIL import Image as PILImage
from io import BytesIO

logger = logging.getLogger("ImageSearchPipeline")

class ImageAnalyzer:
    """Class for analyzing images using vision models"""
    
    def __init__(self, vision_llm):
        self.vision_llm = vision_llm
    
    def encode_image_from_url(self, image_url: str) -> Optional[str]:
        """Encode image from URL to base64 string with improved error handling"""
        try:
            # Add User-Agent header to avoid some 403 errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Check if URL ends with a supported image extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            url_has_valid_extension = any(image_url.lower().endswith(ext) for ext in valid_extensions)
            
            # Skip URLs that aren't likely to be images
            if not url_has_valid_extension and not any(fmt in image_url.lower() for fmt in ['image', 'photo', 'jpg', 'jpeg', 'png']):
                logger.warning(f"URL doesn't appear to be an image, skipping: {image_url}")
                return None
            
            # Special handling for known problematic domains (like Newsweek)
            special_domain_handling = False
            if any(domain in image_url.lower() for domain in ['newsweek.com', 'wrightsvillebeachmagazine.com']):
                logger.info(f"Using special handling for known problematic domain: {image_url}")
                special_domain_handling = True
                
            # Make the request with a timeout
            response = requests.get(image_url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check content type to ensure it's an image
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']):
                logger.warning(f"URL doesn't return a supported image content type: {content_type}")
                return None
                
            # Download image and convert to base64
            image_data = response.content
            
            # For special domains, skip PIL validation and use a different approach
            if special_domain_handling:
                try:
                    # For known problematic domains, use a more direct approach
                    # Use PIL only for resizing if needed, not for validation
                    try:
                        img = PILImage.open(BytesIO(image_data))
                        width, height = img.size
                        
                        # Resize if needed
                        if width > 2000 or height > 2000:
                            logger.info(f"Resizing image from {width}x{height}")
                            if width > height:
                                new_width = min(width, 2000)
                                new_height = min(int(height * (new_width / width)), 2000)
                            else:
                                new_height = min(height, 2000)
                                new_width = min(int(width * (new_height / height)), 2000)
                            
                            img = img.resize((new_width, new_height))
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG", quality=95)
                            image_data = buffer.getvalue()
                    except Exception as resize_error:
                        # If resizing fails, just use the original image data
                        logger.warning(f"Could not resize image (using original): {resize_error}")
                    
                    return base64.b64encode(image_data).decode('utf-8')
                except Exception as special_error:
                    logger.warning(f"Special handling failed for {image_url}: {special_error}")
                    # Continue to regular processing as fallback
            
            # Verify it's a valid image and check specs by trying to open it with PIL
            try:
                # Log response headers for debugging
                logger.debug(f"Response headers for {image_url}: {response.headers}")
                logger.debug(f"Content type: {response.headers.get('content-type')}")
                
                # Create a BytesIO object from the image data
                image_stream = BytesIO(image_data)
                
                try:
                    # Try to open the image
                    img = PILImage.open(image_stream)
                    
                    # Try to validate the image format
                    if not img.format:
                        logger.warning(f"Couldn't determine image format for {image_url}")
                        # Try to identify format from content
                        image_stream.seek(0)
                        header = image_stream.read(16)  # Read first 16 bytes for signature
                        image_stream.seek(0)  # Reset pointer
                        
                        # Manual format detection using file signatures
                        if header.startswith(b'\xff\xd8'):  # JPEG signature
                            logger.debug(f"Manual detection identified JPEG signature")
                            img.format = 'JPEG'
                        elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG signature
                            logger.debug(f"Manual detection identified PNG signature")
                            img.format = 'PNG'
                        elif header.startswith(b'GIF8'):  # GIF signature
                            logger.debug(f"Manual detection identified GIF signature")
                            img.format = 'GIF'
                        elif header.startswith(b'RIFF') and b'WEBP' in header:  # WEBP signature
                            logger.debug(f"Manual detection identified WEBP signature")
                            img.format = 'WEBP'
                    
                    # Check if format is supported by OpenAI
                    if img.format not in ['PNG', 'JPEG', 'WEBP', 'GIF']:
                        logger.warning(f"Image format {img.format} not supported by OpenAI, skipping")
                        return None
                    
                    # Check if this is a CMYK JPEG (common in professional photography)
                    if img.format == 'JPEG' and img.mode == 'CMYK':
                        logger.info(f"Converting CMYK JPEG to RGB for compatibility")
                        # Convert CMYK to RGB which is more widely supported
                        img = img.convert('RGB')
                    
                    # Check if GIF is animated (OpenAI doesn't support animated GIFs)
                    if img.format == 'GIF' and 'duration' in img.info:
                        logger.warning(f"Animated GIF detected, skipping")
                        return None
                    
                    # Check image size limit (20MB)
                    img_size_mb = len(image_data) / (1024 * 1024)
                    if img_size_mb > 20:
                        logger.warning(f"Image size {img_size_mb:.2f}MB exceeds OpenAI's 20MB limit, skipping")
                        return None
                    
                    # Check image dimensions
                    width, height = img.size
                    if width < 50 or height < 50:
                        logger.warning(f"Image too small ({width}x{height}), skipping")
                        return None
                    
                    if width > 2000 or height > 2000:
                        logger.warning(f"Image dimensions ({width}x{height}) exceed OpenAI's recommended maximum, resizing")
                        # Resize to keep within OpenAI's recommendations
                        if width > height:
                            new_width = min(width, 2000)
                            new_height = min(int(height * (new_width / width)), 2000)
                        else:
                            new_height = min(height, 2000)
                            new_width = min(int(width * (new_height / height)), 2000)
                        
                        img = img.resize((new_width, new_height))
                    
                    # Convert to standard format regardless of input
                    buffer = BytesIO()
                    # Always save as JPEG for consistent format
                    img.save(buffer, format="JPEG", quality=95)
                    image_data = buffer.getvalue()
                    
                    return base64.b64encode(image_data).decode('utf-8')
                    
                except Exception as inner_e:
                    logger.warning(f"Error processing image data from {image_url}: {inner_e}")
                    # Try one more approach - force JPEG interpretation
                    try:
                        image_stream.seek(0)
                        img = PILImage.open(image_stream, formats=['JPEG'])
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG")
                        image_data = buffer.getvalue()
                        return base64.b64encode(image_data).decode('utf-8')
                    except Exception as last_e:
                        logger.warning(f"Final attempt failed for {image_url}: {last_e}")
                        return None
                
            except Exception as e:
                logger.warning(f"Invalid image data from URL {image_url}: {e}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error encoding image from URL {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"General error with URL {image_url}: {e}")
            return None
    
    def encode_image_from_file(self, image_path: str) -> Optional[str]:
        """Encode image from file to base64 string"""
        try:
            # Verify it's a valid image first
            PILImage.open(image_path).verify()
            
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image from file {image_path}: {e}")
            return None
    
    def check_elements_presence(self, image_source: str, elements: List[str], is_url: bool = False) -> Dict[str, float]:
        """Check if specific elements are present in the image"""
        try:
            # Encode image based on source type
            if is_url:
                base64_image = self.encode_image_from_url(image_source)
            else:
                base64_image = self.encode_image_from_file(image_source)
            
            if not base64_image:
                logger.warning(f"Failed to encode image from {'URL' if is_url else 'file'}: {image_source}")
                return {}
            
            elements_list = ", ".join([f"'{elem}'" for elem in elements])
            
            prompt_text = f"""
            Carefully analyze this image and determine if each of the following elements is present:
            
            Elements: {elements_list}
            
            For each element, provide a score from 0 to 10:
            - 0: Not present at all
            - 5: Partially present or similar but not exact
            - 10: Clearly present
            
            Format your response as valid JSON like this example:
            {{
              "elements": {{
                "element1": 8,
                "element2": 0,
                "element3": 5
              }}
            }}
            
            IMPORTANT: Ensure your JSON is correctly formatted with double quotes, not single quotes.
            """
            
            messages = [
                SystemMessage(content="You are a precise image analyzer that carefully identifies whether specific elements are present in images."),
                HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            
            # Get evaluation from vision model
            response = self.vision_llm.invoke(messages)
            
            # Extract JSON from response with improved parsing
            try:
                response_text = response.content
                
                # Remove any markdown formatting
                json_content = response_text
                
                # Try to extract JSON block from markdown
                code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if code_block_match:
                    json_content = code_block_match.group(1)
                
                # Ensure we only have the JSON portion
                json_start = json_content.find('{')
                json_end = json_content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = json_content[json_start:json_end]
                    
                    # Fix common JSON formatting issues
                    json_content = json_content.replace("'", '"')  # Replace single quotes with double quotes
                    
                    # Clean up any trailing commas (common JSON error)
                    json_content = re.sub(r',\s*}', '}', json_content)
                    json_content = re.sub(r',\s*]', ']', json_content)
                    
                    # Try to parse the JSON
                    parsed_data = json.loads(json_content)
                    
                    # Extract elements
                    if "elements" in parsed_data:
                        return parsed_data["elements"]
                    else:
                        return parsed_data  # Maybe it's directly the elements dict
                
                # If JSON parsing failed, manual extraction as fallback
                elements_dict = {}
                for element in elements:
                    # Try to find scores like "element: 8" or "element": 8
                    score_match = re.search(rf'["\']?{re.escape(element)}["\']?\s*[:=]\s*(\d+)', response_text)
                    if score_match:
                        elements_dict[element] = float(score_match.group(1))
                    else:
                        elements_dict[element] = 0.0
                
                return elements_dict
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                
                # Last resort fallback - return default values
                return {elem: 0.0 for elem in elements}
        except Exception as e:
            logger.error(f"Error checking elements in image: {e}")
            return {}
    
    def generate_description(self, image_source: str, is_url: bool = False) -> str:
        """Generate detailed description of an image from URL or file path"""
        try:
            # Encode image based on source type
            if is_url:
                base64_image = self.encode_image_from_url(image_source)
            else:
                base64_image = self.encode_image_from_file(image_source)
            
            if not base64_image:
                return f"Error: Unable to process image"
            
            messages = [
                SystemMessage(content="You are an expert at describing images in detail. Describe all important elements, objects, people, settings, colors, and actions in the image."),
                HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail, including all key elements that might be used for searching:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            
            response = self.vision_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating description for {image_source}: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def evaluate_match(self, image_source: str, description_components: Dict[str, Any], is_url: bool = False) -> float:
        """Evaluate how well an image matches given description components"""
        try:
            # Encode image based on source type
            if is_url:
                base64_image = self.encode_image_from_url(image_source)
            else:
                base64_image = self.encode_image_from_file(image_source)
            
            if not base64_image:
                return 0.0
            
            # Create prompt for evaluation
            elements_text = ", ".join([comp["element"] for comp in description_components["elements"]])
            subjects_text = ", ".join(description_components["subjects"])
            
            prompt_text = f"""
            Evaluate how well this image matches the following description components:
            
            Elements: {elements_text}
            Main subjects: {subjects_text}
            Setting: {description_components.get('setting', 'unknown')}
            Action: {description_components.get('action', 'unknown')}
            
            For each component, provide a match score from 0 to 10 (0 = not present at all, 10 = perfect match).
            Finally, provide an overall match percentage from 0 to 100.
            
            Format your response as valid JSON like this example:
            {{
              "element_scores": {{
                "element1": 8,
                "element2": 0
              }},
              "subject_score": 7,
              "setting_score": 5,
              "action_score": 3,
              "overall_percentage": 65
            }}
            
            IMPORTANT: Ensure your JSON is correctly formatted with double quotes, not single quotes.
            """
            
            messages = [
                SystemMessage(content="You are an expert at analyzing how well images match specific descriptions."),
                HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            
            # Get evaluation from vision model
            response = self.vision_llm.invoke(messages)
            
            # Extract JSON from response with improved parsing
            try:
                response_text = response.content
                
                # Remove any markdown formatting
                json_content = response_text
                
                # Try to extract JSON block from markdown
                code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if code_block_match:
                    json_content = code_block_match.group(1)
                
                # Ensure we only have the JSON portion
                json_start = json_content.find('{')
                json_end = json_content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = json_content[json_start:json_end]
                    
                    # Fix common JSON formatting issues
                    json_content = json_content.replace("'", '"')  # Replace single quotes with double quotes
                    
                    # Clean up any trailing commas (common JSON error)
                    json_content = re.sub(r',\s*}', '}', json_content)
                    json_content = re.sub(r',\s*]', ']', json_content)
                    
                    # Try to parse the JSON
                    result = json.loads(json_content)
                    
                    # Return overall percentage as a float between 0 and 1
                    return result.get("overall_percentage", 0) / 100
                
                # If JSON parsing failed, extract percentage using regex
                percentage_match = re.search(r'"?overall_percentage"?["\s:]+(\d+)', response_text)
                if percentage_match:
                    return int(percentage_match.group(1)) / 100
                
                return 0.0  # Default score if parsing fails
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing evaluation response: {e}")
                
                # Fallback: extract percentage using regex
                percentage_match = re.search(r'"?overall_percentage"?["\s:]+(\d+)', response.content)
                if percentage_match:
                    return int(percentage_match.group(1)) / 100
                    
                # Last resort: look for any percentage in the response
                percentage_match = re.search(r'(\d{1,3})%', response.content)
                if percentage_match:
                    return int(percentage_match.group(1)) / 100
                    
                return 0.0  # Default score if all parsing fails
        except Exception as e:
            logger.error(f"Error evaluating match for {image_source}: {e}")
            return 0.0