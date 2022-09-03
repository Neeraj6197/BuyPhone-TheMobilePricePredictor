import streamlit as st
import pickle
import sklearn
from xgboost import XGBRegressor
import numpy as np
from sklearn import *

df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title("Mobile Price Predictor")

#brand
brand_list = df['brand'].unique().tolist()
brand_list.sort()

brand = st.selectbox('Select a brand',brand_list)

#model_type
model_list = df['model'].unique().tolist()
model_list.sort()
model_type = st.selectbox('Select a model',model_list)

#color:
color_list = df['color'].unique().tolist()
color_list.sort()
color = st.selectbox('Select a color',color_list)

#ram
ram_list = df['ram'].unique().tolist()
ram_list.sort()
ram = st.selectbox('RAM(in GB)',ram_list)

#rom
rom_list = df['rom'].unique().tolist()
rom_list.sort()
rom = st.selectbox('ROM(in GB)',rom_list)

#expandable
expandable_list = df['expandable'].unique().tolist()
expandable_list.sort()
expandable = st.selectbox('Expandable Memory(in GB)',expandable_list)

#rear camera:
rear_camera_list = df['rear_camera'].unique().tolist()
rear_camera_list.sort()
rear_camera = st.selectbox('Rear Camera(in MP)',rear_camera_list)

#front camera:
front_camera_list = df['front_camera'].unique().tolist()
front_camera_list.sort()
front_camera = st.selectbox('Front Camera(in MP)',front_camera_list)

#size:
size_list = df['size(inch)'].unique().tolist()
size_list.sort()
size = st.selectbox('Select Display Size(inch)',size_list)

#predictor:
if st.button("Predict Price"):
    query = np.array([brand,model_type,color,ram,rom,expandable,rear_camera,front_camera,size],dtype=object)
    query = query.reshape(1,9)

    st.title("The predicted price for this configuration is " + str(int(np.exp(pipe.predict(query)[0]))) + "/-")

