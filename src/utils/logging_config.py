import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup specific loggers
    pipeline_logger = logging.getLogger("ImageSearchPipeline")
    pipeline_logger.setLevel(level)
    
    return root_logger