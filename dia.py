#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017')
database = client['srilikhitha']


# In[2]:


db = database.get_collection("dia")


# In[3]:


import pandas as pd
df = pd.DataFrame(list(db.find()))


# In[4]:


df


# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df['Outcome'].value_counts()


# In[10]:


df.groupby('Outcome').mean()


# In[18]:


# separating the data and labels
X = df.drop(columns = ['_id','Outcome'], axis=1)
Y = df['Outcome']
X = X.astype(str).astype(float)


# In[19]:


print(X)


# In[20]:


print(Y)


# Train Test Split

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

# In[23]:


classifier = svm.SVC(kernel='linear')


# In[24]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# Accuracy Score

# In[25]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[26]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[27]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[28]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Making a Predictive System

# In[29]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# Saving the trained model

# In[30]:


import pickle


# In[31]:


filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[32]:


# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


# In[33]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[34]:


for column in X.columns:
  print(column)


# In[ ]:




