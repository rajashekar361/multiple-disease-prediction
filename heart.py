#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017')
database = client['srilikhitha']


# In[2]:


db = database.get_collection("heart")


# In[3]:


import pandas as pd
df = pd.DataFrame(list(db.find()))


# In[4]:


df


# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


df.head


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df['target'].value_counts()


# 1 --> Defective Heart
# 
# 0 --> Healthy Heart

# Splitting the Features and Target

# In[19]:


X = df.drop(columns=['_id','target'], axis=1)
Y = df['target']
X = X.astype(str).astype(float)


# In[20]:


print(X)


# In[21]:


print(Y)


# Splitting the Data into Training data & Test Data

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# # Logistic Regression

# In[25]:


model = LogisticRegression()


# In[26]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# Accuracy Score

# In[27]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[28]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[29]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[30]:


print('Accuracy on Test data : ', test_data_accuracy)


# Building a Predictive System

# In[31]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# Saving the trained model

# In[32]:


import pickle


# In[33]:


filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[34]:


# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))


# In[35]:


for column in X.columns:
  print(column)


# In[ ]:




