import os
import sys
import pandas as pd  
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from feast import Field, Feature, FeatureView, FileSource, FeatureStore, Entity
from feast.types import Int32, String, Float32, Int64
from feast.value_type import ValueType
from datetime import datetime, timedelta


@dataclass
class DataTransformationConfig:
  preprocess_obj_file_path = os.path.join("artifacts/data-transformation", "preprocessor.pkl")
  feature_store_repo_path = "feature_repo"
  
class DataTransformation:
  def __init__(self):
    try:
      self.data_transformation_config = DataTransformationConfig()

      # Get absolute path and creating feature store directory structure
      repo_path = os.path.abspath(self.data_transformation_config.feature_store_repo_path)
      logging.info(f"This is repo path {repo_path}")
      os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)

      # creating feature store yaml file with minimal configuration 
      feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
      
      # Feature store configuration
      feature_store_yaml_config = """project: income_prediction
registry: data/registry.db
provider: local
online_store:
  type: sqlite
offline_store:
  type: file
entity_key_serialization_version: 2"""

      # Write the feature store configuration to the yaml file
      with open(feature_store_yaml_path, "w") as f:
        f.write(feature_store_yaml_config)
      logging.info(f"Feature store configuration written to {feature_store_yaml_path}")
      
      # Verify the content of the feature store yaml file
      with open(feature_store_yaml_path, "r") as f:
        logging.info(f"Feature store configuration content:\n{f.read()}")
        
      # Initialize Feature Store
      self.feature_store = FeatureStore(repo_path=repo_path)
      logging.info("Feature store initialized successfully.")
    
    except Exception as e:
      logging.error("Error in initializing {str(e)}")
      raise CustomException(e, sys)
    
  def get_data_transformation_obj(self):
    try:
      logging.info("Data transformation has been started.")
      numerical_features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week', 'native_country']
      
      num_pipeline = Pipeline(
        steps=[
          ('imputer', SimpleImputer(strategy='median')),
          ('scaler', StandardScaler())
        ]
      )
      
      preprocessor = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_features)
      ])
      
      return preprocessor
    
    except Exception as e:
      logging.error("Error in data transformation {str(e)}")
      raise CustomException(e, sys)
    
  def remove_outliers_IQR(self, col, df):
    try:
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1
      
      upper_bound = Q3 + 1.5 * IQR
      lower_bound = Q1 - 1.5 * IQR 
      df.loc[(df[col] > upper_bound), col] = upper_bound
      df.loc[(df[col] < lower_bound), col] = lower_bound
      
      return df 
    except Exception as e:
      logging.error("Error in outlier handling {str(e)}")
      raise CustomException(e, sys)
    
  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_data = pd.read_csv(train_path)
      test_data = pd.read_csv(test_path)
      
      logging.info("Read Train and Test data completed.")
      preprocessing_obj = self.get_data_transformation_obj()
      
      target_column_name = 'income'
      numerical_features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week', 'native_country']
      
      input_feature_train_df = train_data.drop(columns=[target_column_name], axis=1)
      target_feature_train_df = train_data[target_column_name]
      
      input_feature_test_df = test_data.drop(columns=[target_column_name], axis=1)
      target_feature_test_df = test_data[target_column_name]
      logging.info("Applying preprocessing object on training and testing datasets.")
      
      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
      
      logging.info("Starting Feature Store ingestion process.")
      
      # Push data to FEAST Feature Store
      self.push_feature_to_store(train_data, "train")
      logging.info("Push train data to feature store")
      
      self.push_feature_to_store(test_data, "test")
      logging.info("Push test data to feature store") 
      
      train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
      test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
      
      save_object(
        file_path=self.data_transformation_config.preprocess_obj_file_path,
        obj=preprocessing_obj
      ) 
      
      return train_arr, test_arr, self.data_transformation_config.preprocess_obj_file_path
      
    except Exception as e:
      logging.error(f'Error in initiate_data_transformation {str(e)}')
      raise CustomException(e, sys)
  
  def push_feature_to_store(self, df, entity_id):
    try:
      if 'event_timestamp' not in df.columns:
        df['event_timestamp'] = pd.Timestamp.now()
      
      if 'entity_id' not in df.columns:
        df['entity_id'] = range(len(df))
        
        
      data_path = os.path.join(self.data_transformation_config.feature_store_repo_path, "data")
      parquet_path = os.path.join(data_path, f'{entity_id}_features.parquet')
      
      # Ensure this directory exists
      os.makedirs(data_path, exist_ok=True)
      
      df.to_parquet(parquet_path, index=False)
      logging.info(f'Save feature data to {parquet_path} successfully.')
      
      data_source = FileSource(
        path=f"data/{entity_id}_features.parquet",
        timestamp_field="event_timestamp"
      )
      
      # Define entities
      entity = Entity(
        name="entity_id",
        value_type=ValueType.INT64,
        description="Entity ID"
      )
      
      # Define featuresView
      feature_view = FeatureView(
        name=f'{entity_id}_features',
        entities=[entity],
        schema=[
          Field(name="age", dtype=Int64),
          Field(name="workclass", dtype=String),
          Field(name="education_num", dtype=Int64),
          Field(name="marital_status", dtype=String),
          Field(name="occupation", dtype=String),
          Field(name="relationship", dtype=String),
          Field(name="race", dtype=String),
          Field(name="sex", dtype=String),
          Field(name="capital_gain", dtype=Int64),
          Field(name="capital_loss", dtype=Int64),
          Field(name="hours_per_week", dtype=Int64),
          Field(name="native_country", dtype=String)
        ],
        source=data_source,
        online=True
      )
      
      # Apply this configuration to the feature store
      self.feature_store.apply([entity, feature_view])
      logging.info(f'Applied entity and feature view for {entity_id} to the feature store successfully.')
      
      # Materialize features to online store
      self.feature_store.materialize(
        start_date = datetime.now() - timedelta(days=1),
        end_date = datetime.now() + timedelta(days=1)
      )
      
    except Exception as e:
      logging.error(f"Error in push_feature_to_store: {str(e)}")
      raise CustomException(e, sys)
  
  def retrieve_features_from_store(self, entity_id):
    try:
      feature_service_name = f"{entity_id}_features"
      feature_vector = self.feature_store.get_online_features(
        feature_refs=[
          f"{entity_id}_features:age",
          f"{entity_id}_features:workclass",
          f"{entity_id}_features:education_num",
          f"{entity_id}_features:marital_status",
          f"{entity_id}_features:occupation",
          f"{entity_id}_features:relationship",
          f"{entity_id}_features:race",
          f"{entity_id}_features:sex",
          f"{entity_id}_features:capital_gain",
          f"{entity_id}_features:capital_loss",
          f"{entity_id}_features:hours_per_week",
          f"{entity_id}_features:native_country"
        ],
        entity_rows=[{"entity_id": i} for i in range(len(df))]
      ).to_df()
      
      logging.info(f'Retrieve feature for {entity_id}')
      return feature_vector
    
    except Exception as e:
      logging.error(f'Error in retrieve_feature_from_store {str(e)}')
      raise CustomException(e, sys)
    
