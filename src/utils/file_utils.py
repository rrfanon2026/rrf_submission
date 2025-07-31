from pathlib import Path
from typing import Optional

def get_model_dir_name(model_name: str) -> str:
    """
    Convert model name to a directory-safe name by replacing hyphens and dots with underscores.
    
    Args:
        model_name: Original model name (e.g., 'gpt-4o-mini' or 'gemini-2.0-flash')
        
    Returns:
        Directory-safe model name (e.g., 'gpt_4o_mini' or 'gemini_2_0_flash')
    """
    return model_name.replace('-', '_').replace('.', '_')

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    return Path(__file__).resolve().parent.parent

def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"

def get_results_dir() -> Path:
    """Get the results directory."""
    return get_project_root() / "results"

def setup_model_directory(model_name: str, is_results: bool = True) -> Path:
    """
    Create and return a model-specific directory.
    
    Args:
        model_name: Name of the model
        is_results: If True, creates directory in results/, otherwise in data/
        
    Returns:
        Path to the model-specific directory
    """
    base_dir = get_results_dir() if is_results else get_data_dir()
    model_dir = base_dir / model_name.replace("-", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_default_filename(model_name: str, prefix: str) -> str:
    """Generate default filename for model outputs."""
    sanitized_model = model_name.replace("-", "_")
    return f"{prefix}_{sanitized_model}.csv"

def setup_logging(model_name: str):
    """
    Set up logging configuration for the given model.
    
    Args:
        model_name: Name of the model to use in log file name
        
    Returns:
        Logger instance configured for the model
    """
    import logging
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{model_name}_{timestamp}.log"
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 