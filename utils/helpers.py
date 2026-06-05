import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a configured logger.
    
    Args:
        name (str): Name of the logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def safe_divide(numerator: float, denominator: float, default_value: float = 0.0) -> float:
    """
    Safely divides two numbers avoiding division by zero.
    
    Args:
        numerator (float): The numerator.
        denominator (float): The denominator.
        default_value (float): Value to return if denominator is 0.
        
    Returns:
        float: The result of division or default_value.
    """
    if denominator == 0:
        return default_value
    return numerator / denominator

if __name__ == "__main__":
    logger = setup_logger("test_logger")
    logger.info("Logger setup successfully.")
    print(f"10 / 2 = {safe_divide(10, 2)}")
    print(f"10 / 0 = {safe_divide(10, 0)}")
