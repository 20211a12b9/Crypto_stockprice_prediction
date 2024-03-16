# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:58:23 2024

@author: chowd
"""

import numpy as np
import pickle
import streamlit as st
from xgboost import XGBRegressor

# Load the XGBoost model
load_data = pickle.load(open('C:/Users/chowd/Downloads/crypto.sav', 'rb'))

def clean_input(input_data):
    # Convert input data to string and remove Unicode characters
    cleaned_data = [str(item).encode('ascii', 'ignore').decode() for item in input_data]
    return cleaned_data

def crypto_price_prediction(input_data):
    input_data_np = np.asarray(input_data, dtype=np.float32)
    input_data_reshaped = input_data_np.reshape(1, -1)
    xg_prediction = load_data.predict(input_data_reshaped)
    return xg_prediction[0]

def main():
    st.title('Crypto Price Prediction')
    
    # Input fields
    names = st.text_input('Name')
    change = st.text_input('Change')
    pchange = st.text_input('Percentage Change')
    volume_in_currency = st.text_input('Volume in Currency (Since 0:00 UTC)')
    volume_in_currency_24hr = st.text_input('Volume in Currency (24Hr)')
    total_volume_all_currencies_24hr = st.text_input('Total Volume All Currencies (24Hr)')
    circulating_supply = st.text_input('Circulating Supply')
    
   
    # Predict price button
    if st.button('Predict Price'):
        input_data = [names, change, pchange, volume_in_currency, volume_in_currency_24hr,
                      total_volume_all_currencies_24hr, circulating_supply]
        cleaned_input = clean_input(input_data)
        price_prediction = crypto_price_prediction(cleaned_input)
        st.success(f'Predicted Price: {price_prediction}')

if __name__ == '__main__':
    main()
