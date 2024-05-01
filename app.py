import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

st.title('RSaid')
datos = pd.read_csv('hprice.csv')
print(datos.head())