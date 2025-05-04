# ui/streamlit_app.py
import os
import sys
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.pipeline import ImageSearchPipeline
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()

def create_streamlit_app():
    """Create and run the Streamlit app"""
    st.title("AI Image Search Pipeline")
    
    # Setup sidebar for configuration
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serpapi_key = st.sidebar.text_input("SerpAPI Key", type="password")
    
    # Main content area
    st.header("Search for Images")
    
    # Predefined examples
    example_descriptions = [
        "Elon Musk standing in a grass field holding an umbrella with a circus in the background",
        "A golden retriever wearing sunglasses riding a skateboard on a beach at sunset",
        "An astronaut playing chess with a robot on top of a mountain",
        "A Victorian-era woman using a modern smartphone in a library"
    ]
    
    example_option = st.selectbox(
        "Select an example or enter your own description below:",
        ["Custom"] + example_descriptions
    )
    
    if example_option == "Custom":
        description = st.text_area("Enter detailed image description")
    else:
        description = example_option
        st.text_area("Description", value=description, disabled=True)
    
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
    
    search_button = st.button("Search Images")
    
    # Cache the pipeline to avoid reinitializing on every search
    @st.cache_resource
    def get_pipeline(openai_key, serp_key):
        return ImageSearchPipeline(openai_key, serp_key)
    
    # Initialize and run search when button is clicked
    if search_button and openai_api_key and serpapi_key and description:
        try:
            with st.spinner("Searching for images that match your description..."):
                pipeline = get_pipeline(openai_api_key, serpapi_key)
                results = pipeline.search(description, top_k=top_k)
            
            # Display results
            if results:
                st.header("Search Results")
                for i, result in enumerate(results):
                    with st.container():
                        st.subheader(f"Result #{i+1} - Match Score: {result.match_score:.2f}")
                        
                        cols = st.columns([2, 3])
                        with cols[0]:
                            # Display image - prefer local path if available, else use URL
                            try:
                                if result.local_path:
                                    image = Image.open(result.local_path)
                                    st.image(image, caption=f"Match Score: {result.match_score:.2f}")
                                else:
                                    st.image(result.url, caption=f"Match Score: {result.match_score:.2f}")
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                                st.image(result.thumbnail, caption="Thumbnail")
                        
                        with cols[1]:
                            st.write(f"**Title:** {result.title}")
                            st.write(f"**Source:** [Visit Page]({result.source_page})")
                            st.write(f"**Dimensions:** {result.width}x{result.height}")
                            st.write(f"**Match Score:** {result.match_score:.2f}")
                            st.write(f"**Image URL:** [Open Image]({result.url})")
                            if result.element_scores:
                                st.write("**Element Scores:**")
                                for element, score in result.element_scores.items():
                                    status = "✅" if score >= 5 else "❌"
                                    st.write(f"  {status} {element}: {score:.1f}/10")
                            
                        st.markdown("---")
            else:
                st.warning("No matching images found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()