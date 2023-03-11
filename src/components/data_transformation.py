import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for Data transformation.
        '''
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 
                                    'race_ethnicity', 
                                    'parental_level_of_education', 
                                    'lunch', 
                                    'test_preparation_course']
            
            numerical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('numerical_pipeline', numerical_pipeline, numerical_features),
                ('categorical_pipeine', categorical_pipeline, categorical_features)
                ]
            )

            for type in ['Numerical', 'Categorical']:
                logging.info(f'{type} features preprocessing completed')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test datasets completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            numerical_features = ['reading_score', 'writing_score']

            input_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]

            input_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_feature_test_arr = preprocessing_obj.transform()
        except Exception as e:
            raise CustomException(e, sys)