import streamlit as st
import numpy as np
import pandas as pd
import pickle

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

valores=np.array([[val_1,val_2,val_3,val_4,val_5]])

precio=model.predict(valores)
st.write(precio)