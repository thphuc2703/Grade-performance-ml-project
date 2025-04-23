import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from exception import Custom_Exception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise Custom_Exception(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try: 
        report = {}
        for name, model in models.items():
            param_grid = params.get(name, {})
            
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model
                
            y_test_ped = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_ped)
            report[name] = test_model_score
            
        return report
    except Exception as e:
       raise Custom_Exception(e, sys)
   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e, sys)    