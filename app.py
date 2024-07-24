import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
# import pickle

# import sklearn
# import the model
pipe = joblib.load('pipe.sav')
df = joblib.load('df.sav')

st.title("LAPTOP PRICE PREDICTOR")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM(IN GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = float(st.number_input('Weight'))
# touchscreen
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS display
ips = st.selectbox('IPS', ['Yes', 'No'])

# PPI
# 1- Screen size
screen_size = st.number_input('Screen Size')
# 2- resolution
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1600', '2560x1440',
                           '2304x1440'])

# CPU
cpu = st.selectbox('CPU ', df['cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)]', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB', [0, 8, 128, 256, 512, 1024, 2048])

# GPU
gpu = st.selectbox('Gpu', df['Gpu'].unique())

# OS
os = st.selectbox('OS', df['OpSys'].unique())

# button
if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** .5 / screen_size
    query = np.array([company, type, ram, gpu, os, weight, touchscreen, ips , ppi, cpu, hdd, ssd])
    query = query.reshape(1, 12)
    st.title("This laptop price is: " + str(int(np.exp(pipe.predict(query)[0])))+" \u20B9 ")
    for x in query:
        print(x)
    print(np.exp(pipe.predict(pd.DataFrame([['HP','Notebook',4,'Intel','Windows',1.49,0,1,165.632118,'Intel Core i5',500,0]],columns=['Company','TypeName','Ram','Gpu','OpSys','Weight','touchscreen','IPS','PPI','cpu brand','HDD','SSD']))))
