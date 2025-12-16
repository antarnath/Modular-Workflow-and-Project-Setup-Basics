import sys
import src.logger as logger
from src.exception import CustomException


if __name__ == "__main__":
  try:
    logging = logger.logging
    logging.info("Starting custom exception testing.")
    
    a = 5
    b = 0
    
    logging.info(f"Attempting division operation with a = {a} and b = {b}")
    result = a / b  # This will raise a ZeroDivisionError
    
    logging.info(f"Division result is {result}")
  except Exception as e:
    logging.error("An exception occurred during division operation.", exc_info=True)
    raise CustomException(e, sys)