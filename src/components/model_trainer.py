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
from sklearn.metrics import r2_score, mean_squared_error
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
            
            params = {
                "Linear Regression": {},
                "Lasso": {
                    'alpha': [0.01, 0.1, 1.0, 10]
                },
                "Ridge": {
                    'alpha': [0.01, 0.1, 1.0, 10]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 10, 20]
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }

            
            model_report, trained_models = evaluate_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test, models=models, params=params)
            
            best_model_score = min(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = trained_models[best_model_name]
            
            
            if best_model_score < 0.6:
                raise Custom_Exception("No best model found")
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            y_predicted = best_model.predict(X_test)
            
            # r2_square = r2_score(y_test, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
            
            return rmse
        
        except Exception as e:
            raise Custom_Exception(e, sys)
            
        
