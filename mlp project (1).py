#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


df=pd.read_csv("tesla.csv")


# In[6]:


df.head()


# In[7]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[8]:


plt.figure(figsize=(16,8))
plt.title('netflix')
plt.xlabel('days')
plt.ylabel('Close price usd ($)')
plt.plot(df['Close'])
plt.show()


# In[9]:


df = df[['Close']]
df.head(4)


# In[10]:


future_days = 25 
df['Prediction'] = df[['Close']].shift(-future_days)
df.head(4)


# In[11]:


x = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(x)


# In[12]:


y =np.array(df['Prediction'])[:-future_days]
print(y)


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[14]:


tree=DecisionTreeRegressor().fit(x_train,y_train)
lr=LinearRegression().fit(x_train,y_train)


# In[15]:


x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[16]:


tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# In[17]:


predictions = lr_prediction

valid = df[x.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('days')
plt.ylabel('close price usd($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[18]:


predictions = tree_prediction

valid = df[x.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('days')
plt.ylabel('close price usd($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[19]:


tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()


# In[ ]:





# In[ ]:





# In[ ]:




