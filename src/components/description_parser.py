import json
import logging
import re
from typing import Dict, Any, List

from langchain.prompts import PromptTemplate

logger = logging.getLogger("ImageSearchPipeline")

class DescriptionParser:
    """Class for parsing and breaking down complex image descriptions"""
    
    def __init__(self, text_llm):
        self.text_llm = text_llm
    
    def break_down_description(self, description: str) -> Dict[str, Any]:
        """Break down a complex image description into searchable components"""
        try:
            # Prompt to break down the description
            template = """
            Break down this complex image description into key elements that could be used for searching.
            For each element, assign an importance score from 1-10 (10 being most important).
            
            Description: {description}
            
            Output the result as a JSON object with the following format:
            {{
                "elements": [
                    {{"element": "element1", "importance": score}},
                    {{"element": "element2", "importance": score}},
                    ...
                ],
                "subjects": ["main subject1", "main subject2"],
                "setting": "overall setting",
                "action": "main action"
            }}
            """
            
            prompt = PromptTemplate(template=template, input_variables=["description"])
            
            # Using invoke instead of run (deprecated)
            result = self.text_llm.invoke(prompt.format(description=description))
            
            # Parse the result
            try:
                # Try to find JSON in a code block
                json_match = re.search(r'``[(?:json)?\s*({.*?})\s*](cci:1://file:///Users/escobarsmacbook/Workspace/trial_docs_upwork/src/components/description_parser.py:12:4-13:32)``', result.content, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire content
                    json_data = json.loads(result.content)
                
                return json_data
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, extract information manually (simplified fallback)
                logger.warning(f"Failed to parse JSON from LLM response. Using fallback parsing.")
                return {
                    "elements": [{"element": description, "importance": 10}],
                    "subjects": [description.split()[0] if description.split() else "unknown"],
                    "setting": "unknown",
                    "action": "unknown"
                }
        except Exception as e:
            logger.error(f"Error breaking down description: {e}")
            # Return basic fallback
            return {
                "elements": [{"element": description, "importance": 10}],
                "subjects": ["unknown"],
                "setting": "unknown",
                "action": "unknown"
            }
    
    def generate_search_queries(self, description_components: Dict[str, Any]) -> List[str]:
        """Generate search queries from description components"""
        queries = []
        
        # Add all elements as individual queries
        for comp in description_components["elements"]:
            queries.append(comp["element"])
        
        # Add subjects as a combined query
        if description_components.get("subjects"):
            queries.append(" ".join(description_components["subjects"]))
        
        # Add setting and action if available
        setting = description_components.get("setting")
        if setting and setting != "unknown":
            queries.append(setting)
        
        action = description_components.get("action")
        if action and action != "unknown":
            queries.append(action)
        
        return queries