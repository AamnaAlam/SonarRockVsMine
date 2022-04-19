#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np


# In[57]:


df=pd.read_csv(r"C:\Users\Dell\Desktop\data1\sonar.all-data.csv")
df


# In[58]:


df.columns


# In[59]:


df.isnull().sum()


# In[60]:


df.dtypes


# In[61]:


df.info


# In[62]:


df.info()


# In[63]:


df.describe()


# In[64]:


x=df.drop(['Label'],axis='columns')
x


# In[65]:


y=df.Label
y


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[68]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)


# In[70]:


inputd=(0.0134,0.0172,0.0178,0.0363,0.0444,0.0744,0.08,0.0456,0.0368,0.125,0.2405,0.2325,0.2523,0.1472,0.0669,0.11,0.2353,0.3282,0.4416,0.5167,0.6508,0.7793,0.7978,0.7786,0.8587,0.9321,0.9454,0.8645,0.722,0.485,0.1357,0.2951,0.4715,0.6036,0.8083,0.987,0.88,0.6411,0.4276,0.2702,0.2642,0.3342,0.4335,0.4542,0.396,0.2525,0.1084,0.0372,0.0286,0.0099,0.0046,0.0094,0.0048,0.0047,0.0016,0.0008,0.0042,0.0024,0.0027,0.0041)
inputdnumpy=np.asarray(inputd)
inputdreshape=inputdnumpy.reshape(1,-1)


# In[71]:


prediction=lr.predict(inputdreshape)
print(prediction)


# In[72]:


if(prediction=='R'):
    print("it is rock")
else:
    print("it is mine")


# In[ ]:




