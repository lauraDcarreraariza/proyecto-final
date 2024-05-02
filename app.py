import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
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

st.title('RSaid')
datos = pd.read_csv('hprice.csv')

with open("model.pickle", "rb") as f:
     model = pickle.load(f)

st.subheader('assess')
val_1=st.slider('Seleccione el valor',
          datos['assess'].min(),
          datos['assess'].max())
st.subheader('bdrms')
val_2=st.slider('Seleccione el valor',
          datos['bdrms'].min(),
          datos['bdrms'].max())
st.subheader('lotsize')
val_3=st.slider('Seleccione el valor',
          datos['lotsize'].min(),
          datos['lotsize'].max())
st.subheader('sqrft')
val_4=st.slider('Seleccione el valor',
          datos['sqrft'].min(),
          datos['sqrft'].max())
st.subheader('colonial')
val_5_selected=st.radio("¿Su casa es colonial?", ('Sí', 'No'))
val_5 = 1 if val_5_selected == 'Sí' else 0

with open("model.pickle", "rb") as f:
     model = pickle.load(f)
valores=np.array([[val_1,val_2,val_3,val_4,val_5]])

precio=model.predict(valores)
st.write(precio)