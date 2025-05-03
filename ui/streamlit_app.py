import os
import sys
import streamlit as st
from PIL import Image

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
    image_folder = st.sidebar.text_input("Image Folder Path", value="./images")
    
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
    def get_pipeline(api_key, folder):
        return ImageSearchPipeline(api_key, folder)
    
    # Initialize and run search when button is clicked
    if search_button and openai_api_key and image_folder and description:
        try:
            # Validate inputs
            if not os.path.exists(image_folder):
                st.error(f"Image folder path does not exist: {image_folder}")
                return
                
            with st.spinner("Initializing pipeline and searching for images..."):
                pipeline = get_pipeline(openai_api_key, image_folder)
                results = pipeline.search(description, top_k=top_k)
            
            # Display results
            if results:
                st.header("Search Results")
                for i, result in enumerate(results):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        try:
                            image = Image.open(result.path)
                            st.image(image, caption=f"Match Score: {result.match_score:.2f}")
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    with col2:
                        st.subheader(f"Result #{i+1}")
                        st.write(f"Filename: {result.filename}")
                        st.write(f"Match Score: {result.match_score:.2f}")
                        st.write(f"Vector Similarity: {result.similarity:.2f}")
            else:
                st.warning("No matching images found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()