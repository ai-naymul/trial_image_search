# AI Image Search Pipeline

A sophisticated image search system that finds the best matches for highly specific image descriptions using LangChain and advanced AI techniques.

## Project Overview

This system addresses the challenge of finding existing images that match complex, multi-condition descriptions. Instead of generating new images, it uses AI to intelligently find and rank the closest available matches when perfect images don't exist.

### Key Features

- Processes highly specific image descriptions and breaks them down into searchable components
- Uses embeddings and similarity search to find candidate images
- Evaluates images for precise matching using vision AI
- Implements a scoring system that weighs different aspects of descriptions
- Provides a user-friendly Streamlit interface

## Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Description Parser**: Breaks down complex descriptions into searchable components
2. **Vector Store Manager**: Handles embeddings and similarity search
3. **Image Analyzer**: Processes images and evaluates matches
4. **Pipeline Orchestrator**: Coordinates the entire search process
5. **UI Component**: Provides an interface for user interaction

## Prerequisites

- Python 3.8+
- OpenAI API key
- A directory containing images to search

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

### Running the UI

```bash
streamlit run ui/streamlit_app.py
```

## Using as a Library

```python
from src.components.pipeline import ImageSearchPipeline

# Initialize the pipeline
pipeline = ImageSearchPipeline(openai_api_key="your_api_key", image_folder_path="./images")

# Search for images
results = pipeline.search("Your detailed image description", top_k=3)

# Process results
for result in results:
    print(f"Filename: {result.filename}")
    print(f"Match Score: {result.match_score}")

```


## Required API Keys

- OpenAI API key

## Design Decision

### Handling Partial Matches
The pipeline implements a two-stage search strategy:

- Broad Search: Uses embeddings to find semantically relevant candidates
- Precise Evaluation: Analyzes candidates with vision AI to score specific attributes

