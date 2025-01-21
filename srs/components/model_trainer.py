import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from srs.exception import CustomException
from srs.logger import logging
from srs.utils import save_object,evaluate_model
from sklearn.linear_model import Lasso,Ridge
from sklearn.svm import SVR
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "linear Regression":LinearRegression(),
                "lasso":Lasso(),
                "ridge":Ridge(),
                "k-nearest":KNeighborsRegressor(),
                "decision_tree":DecisionTreeRegressor(),
                "rf":RandomForestRegressor(),
                "adaboost":AdaBoostRegressor(),
                "Support Vector Regressor":SVR(),
                "catboost":CatBoostRegressor(verbose=False),
                "XGBRegressor":XGBRegressor(),
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"find out the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted_output=best_model.predict(X_test)
            r2_value=r2_score(y_test,predicted_output)
            return r2_value
        
        except Exception as e:
            raise CustomException(e,sys)
