import sys
from src.logger import logging

class CustomException(Exception):
  def __init__(self, error_message: str, error_detail: sys):
    super().__init__(error_message)
    self.error_message = self._generate_detailed_error_message(error_message, error_detail)
  
  
  def _generate_detailed_error_message(self, error_message: str, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    detailed_message = (
      f'\nError occurred in python script:'
      f'\n→ File: {file_name}'
      f'\n→ Line number: {exc_tb.tb_lineno}'
      f'\n→ Error message: {error_message}'
    )
    
    logging.error(detailed_message)
    return detailed_message
  
  def __str__(self) -> str:
    return self.error_message