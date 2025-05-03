import base64
import json
import logging
import re
from typing import Dict, Any

from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger("ImageSearchPipeline")

class ImageAnalyzer:
    """Class for analyzing images using vision models"""
    
    def __init__(self, vision_llm):
        self.vision_llm = vision_llm
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def generate_description(self, image_path: str) -> str:
        """Generate detailed description of an image"""
        try:
            base64_image = self.encode_image(image_path)
            
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
            logger.error(f"Error generating description for {image_path}: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def evaluate_match(self, image_path: str, description_components: Dict[str, Any]) -> float:
        """Evaluate how well an image matches given description components"""
        try:
            base64_image = self.encode_image(image_path)
            
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
            
            Format your response as a JSON object:
            {{
                "element_scores": {{
                    "element1": score,
                    "element2": score,
                    ...
                }},
                "subject_score": score,
                "setting_score": score,
                "action_score": score,
                "overall_percentage": percentage
            }}
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
            
            # Try to extract JSON from the response
            try:
                # First try to find JSON block in code formatting
                json_match = re.search(r'``[(?:json)?\s*({.*?})\s*](cci:1://file:///Users/escobarsmacbook/Workspace/trial_docs_upwork/src/components/description_parser.py:12:4-13:32)``', response.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Otherwise try to extract anything that looks like a JSON object
                    json_match = re.search(r'({[\s\S]*?})', response.content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response.content
                
                result = json.loads(json_str)
                # Return overall percentage as a float between 0 and 1
                return result.get("overall_percentage", 0) / 100
            except json.JSONDecodeError:
                # Fallback: extract percentage using regex
                percentage_match = re.search(r'overall_percentage["\s:]+(\d+)', response.content)
                if percentage_match:
                    return int(percentage_match.group(1)) / 100
                return 0.0  # Default score if parsing fails
        except Exception as e:
            logger.error(f"Error evaluating match for {image_path}: {e}")
            return 0.0