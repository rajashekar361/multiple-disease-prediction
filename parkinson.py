#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017')
database = client['srilikhitha']


# In[2]:


db = database.get_collection("data'")


# In[3]:


import pandas as pd
df = pd.DataFrame(list(db.find()))


# In[59]:


import numpy as np
import pandas as pd


# In[4]:


df


# In[5]:


df.head


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[26]:


df.groupby('status').mean(6)


# Data PreProcessing

# In[43]:


X = df.drop(columns=['_id','name','status'], axis=1)
Y = df['status']
X = X.astype(str).astype(float)


# In[44]:


print(X)


# In[45]:


print(Y)


# Splitting the data to training data & Test data

# In[46]:


import sklearn
from sklearn.model_selection  import train_test_split


# In[47]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[48]:


print(X.shape, X_train.shape, X_test.shape)


# In[49]:


print(X_train,Y_train)


# # Support Vector Machine Model

# In[50]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[51]:


model = svm.SVC(kernel='linear')


# In[53]:


# training the SVM model with training data
model.fit(X_train, Y_train)


# Accuracy

# In[54]:


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[55]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[56]:


# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[57]:


print('Accuracy score of test data : ', test_data_accuracy)


# In[60]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")


# In[61]:


import pickle


# In[62]:


filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[63]:


# loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# In[64]:


for column in X.columns:
  print(column)


# In[ ]:





# In[ ]:




