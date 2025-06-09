import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import Windows fix utilities
from utils.windows_fix import apply_windows_asyncio_fixes

# Use direct import
from ui.app import create_app

def main():
    """Main entry point"""
    logger.info("Starting YouTube Shorts Generator")
    
    # Apply Windows-specific fixes at the beginning
    if os.name == 'nt':  # Windows
        apply_windows_asyncio_fixes()
    
    # Create and launch the Gradio app in local-only mode
    app = create_app(use_modal=False)  # Set to use local processing only
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()
