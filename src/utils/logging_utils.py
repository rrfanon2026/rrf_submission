import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger that writes to both console and file.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_rules_generation(logger: logging.Logger, 
                        stage: str, 
                        rules: str, 
                        provider: str,
                        model: str) -> None:
    """
    Log rules generation with consistent formatting.
    
    Args:
        logger: Logger instance
        stage: Stage of generation (initial/validation)
        rules: Generated rules
        provider: LLM provider name
        model: Model name
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"{stage.upper()} RULES GENERATION - {provider.upper()} ({model})")
    logger.info(f"{'='*80}\n")
    logger.info(rules)
    logger.info(f"\n{'='*80}\n") 