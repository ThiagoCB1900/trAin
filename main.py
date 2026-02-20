"""
trAIn Health - Clinical ML Studio
==================================
Main entry point for the application.

This is a production-ready ML experimentation platform for healthcare
with emphasis on reproducibility and governance.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_health.log"),
    ],
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Launch the trAIn Health application.
    
    Initializes the PyQt6 GUI and starts the main event loop.
    """
    logger.info("Starting trAIn Health - Clinical ML Studio")
    
    try:
        # Import GUI after logging is configured
        from main_gui import run_application
        
        logger.info("Initializing GUI...")
        run_application()
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
