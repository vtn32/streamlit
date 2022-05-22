
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
from datetime import date
start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter Stock Ticker','AAPL')



yf.pdr_override()
df = data.get_data_yahoo(user_input,start,end)

#Describing the data
st.subheader('Data from 2010 to today')
st.write(df.describe())


#Visualization

st.subheader('Time Series Data of Closing Price')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


ma100 = df.Close.rolling(100).mean() #calculating the moving average (MA) for 100 days
ma200 = df.Close.rolling(200).mean() #calculating the moving average (MA) for 200 days

st.subheader('Closing Price vs Time Chart with MA100')
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,label = 'MA100')
plt.plot(df.Close,label = 'Close Price')
plt.legend(loc = 'upper left')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA100 and MA200')
fig = plt.figure(figsize = (12,6))
plt.plot(ma200, label = 'MA200')
plt.plot(ma100,label = 'MA100')
plt.plot(df.Close, label = 'Close Price')
plt.legend(loc = 'upper left')
st.pyplot(fig)

#  Data Partitioning into training and testing set with a ratio of 70/30 respectively. Target Variable would be the Close price.

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)]) 
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))]) 
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

# Data partitioning into xtrain and ytrain
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]): #index the number of rows of the training set
  x_train.append(data_training_array[i-100:i]) #using the previous 99 close price to predict the #100 close price
  y_train.append(data_training_array[i,0]) 
x_train, y_train = np.array(x_train), np.array(y_train)

# Load Model
model = load_model('keras_model.h5')

# Testing
past_100_days = data_training.tail(100) #retreive the last 100 days before index #1760 to predict indext #1760
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df) #scale the testing data 

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test *scale_factor


#Final Graph
st.subheader('Prediction vs Actual Price')
fig2= plt.figure(figsize = (12,6))
plt.plot(y_test , 'b',label = 'Actual Testing Price')
plt.plot(y_predicted, 'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
