import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
  raw_data_path: str = os.path.join('artifacts/data_ingestion', 'raw.csv')
  train_data_path: str = os.path.join('artifacts/data_ingestion', 'train.csv')
  test_data_path: str = os.path.join('artifacts/data_ingestion', 'test.csv')
  
class DataIngestion:
  def __init__(self):
    self.ingestion_config = DataIngestionConfig()
  
  def initiate_data_ingestion(self):
    logging.info("Data Ingestion has started")
    try:
      logging.info("Data reading using pandas from local file system")
      data = pd.read_csv(os.path.join('data-source', 'income_cleandata.csv'))
      logging.info("Data reading has been completed")
      
      os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
      data.to_csv(self.ingestion_config.raw_data_path, index=False)
      logging.info("Raw data store successfully")
      
      train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
      logging.info("Raw data has been split into train and test data set!!")
      
      train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
      logging.info("train data store successfully")
      
      test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
      logging.info("Test data store successfully")
      logging.info("Data Ingestion is completed")
      
      return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
      
    except Exception as e:
      logging.info("Error occured in data ingestion stage")
      raise CustomException(e, sys)
    
    
if __name__ == "__main__":
  data_ingestion = DataIngestion()
  train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
  
  data_transformation = DataTransformation()
  train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
  