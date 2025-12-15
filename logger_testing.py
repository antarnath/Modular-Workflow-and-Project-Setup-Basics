from src.logger import logging

def main():
  try:
    logging.info("Starting the main function.")
    logging.info("Step - 1: Initializing Application")
    
    x = 5
    y = 10
    
    logging.info(f"Step -2: Performing addition operation with value x = {x} and y = {y}")
    result = x + y
    
    logging.info(f"Step - 3: Addition result is {result}")
    
    if result > 10:
      logging.warning(f"Result {result} is greater than threshold (10)")
      
  except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise e

if __name__ == "__main__":
  logging.info("="*50)
  logging.info("Application Execution Started")
  logging.info("="*50)
  
  try:
    main()
    logging.info("Application executed successfully.")
  except Exception as e:
    logging.error("Application run failed.", exc_info=True)
  finally:
    logging.info("="*50)
    logging.info("Application Execution Ended")
    logging.info("="*50)