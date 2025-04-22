import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from exception import Custom_Exception
from logger import logging
from dataclasses import dataclass
from utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifact", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise Custom_Exception("No best model found")
            logging.info("Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            y_predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, y_predicted)
            
            return r2_square
        
        except Exception as e:
            raise Custom_Exception(e, sys)
            
        
