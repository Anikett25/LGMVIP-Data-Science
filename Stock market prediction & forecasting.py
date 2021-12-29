#!/usr/bin/env python
# coding: utf-8

# # LGMVIP Data Analytics
# 
# ## Task 3- Prediction and Forecasting using LSTM method
# 
# ### Name: Aniket Taru

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[2]:


#loading data
Data = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"
df = pd.read_csv(Data)
df.head()


# In[3]:


#Sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by = 'Date')
df.head()


# In[4]:


plt.plot(df['Close'])


# In[5]:


dfclose = df['Close']


# In[6]:


#Preparing data
scaler = MinMaxScaler(feature_range=(0,1))
dfclose = scaler.fit_transform(np.array(dfclose).reshape(-1,1))


# In[7]:


#Splitting dataset in train and test data
trset = int(len(dfclose)*0.70)
ttset = len(dfclose)-trset

train,test = dfclose[0:trset,:],dfclose[trset:len(dfclose),:1]


# In[8]:


#converting array values calculated into matrix
def mat(ds,time_step = 1):
    dataX, dataY = [],[]
    for i in range(len(ds)-time_step-1):
        a = ds[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(ds[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[9]:


time_step = 100
x_train,y_train = mat(train,time_step)
x_test,y_test = mat(test,time_step)


# In[10]:


x_train.shape


# In[11]:


y_train.shape


# In[12]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)


# **LSTM model of data**

# In[13]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[14]:


model.fit(x_train, y_train, validation_split=0.1, epochs=60, batch_size=64, verbose=1)


# In[15]:


#Predictions of the model
pred = model.predict(x_test)

#performing inverse transformation on predictions
inv_pred = scaler.inverse_transform(pred)


# In[16]:


inv_pred


# In[17]:


#checking mean squared error of the model
mse = math.sqrt(mean_squared_error(y_test,pred))
print("The Mean squared error of the model is: ", mse)


# In[18]:


temp_input = list(x_test)
temp_input = temp_input[0].tolist()


# In[19]:


temp_input

