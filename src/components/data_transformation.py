import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exception import Custom_Exception
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_obj(self):
        '''
        This function create to transform data
        '''
        try:
            numerical_column = ['reading score', 'writing score']
            categorical_column = [
                'gender', 
                'race/ethnicity', 
                'parental level of education', 
                'lunch', 
                'test preparation course'
            ]
            num_pipeline = Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy="median")),# fill missing value with media
                ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Numerical columns: {numerical_column}')
            logging.info(f'Categorical columns: {categorical_column}')
            
            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipline", num_pipeline, numerical_column),
                    ("cat_pipline", cat_pipeline, categorical_column)
                ]
            )
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test successfully")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_obj()
            
            target_col_name = "math score"
            numerical_column = ['reading score', 'writing score']
            
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )
            
        except Exception as e:
            raise Custom_Exception(e, sys)