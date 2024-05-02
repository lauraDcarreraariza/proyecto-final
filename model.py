import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy as scp
import optuna
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

datos = pd.read_csv('hprice.csv')
print(datos.head())
    
y = datos['price']
X = datos.drop(columns='price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.4,
    'colsample_bytree': 0.4,
    'reg_alpha': 0,
    'reg_lambda': 1
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Mejor RMSE:", rmse)

with open("model.pickle", "wb") as f:
    pickle.dump(model, f)