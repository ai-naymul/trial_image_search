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
    
    def generate_search_queries(self, description_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate structured search queries from description components"""
        # Sort elements by importance
        sorted_elements = sorted(
            description_components["elements"], 
            key=lambda x: x.get("importance", 0),
            reverse=True
        )
        
        # Generate combinations of essential elements
        core_elements = [elem["element"] for elem in sorted_elements[:3]]  # Take top 3 elements
        
        queries = []
        
        # First query: all key elements in one search
        all_elements_query = " ".join(core_elements)
        queries.append({
            "query": all_elements_query,
            "weight": 1.0,
            "elements": core_elements
        })
        
        # Generate more specific queries with essential combinations
        for i, elem1 in enumerate(core_elements):
            for j, elem2 in enumerate(core_elements[i+1:], i+1):
                if i != j:
                    query = f"{elem1} {elem2}"
                    queries.append({
                        "query": query,
                        "weight": 0.7,
                        "elements": [elem1, elem2]
                    })
        
        # Add individual essential elements
        for i, elem in enumerate(core_elements):
            queries.append({
                "query": elem,
                "weight": 0.5,
                "elements": [elem]
            })
        print("Here is the query:", queries)
        return queries