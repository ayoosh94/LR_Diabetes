#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# In[28]:


# Load the pickled lr object
with open('logistic_regression_model.pkl', 'rb') as lr_file:
    scaler, lr = pickle.load(lr_file)


# In[41]:


# Creating new unknown data for input variables, using numpy. 
#The variables are in the range of min and max of these variables
#generating 154 random values, to match the Test cases 



unknown_data = {
    
    'Pregnancies': np.random.randint(1, 17, size=154),
    'Glucose': np.random.randint(44, 199, size=154),
    'BloodPressure': np.random.randint(24, 122, size=154),
    'SkinThickness': np.random.randint(7, 99, size=154),
    'Insulin': np.random.randint(18, 846, size=154),
    'BMI': np.random.randint(18, 68, size=154),
    'DiabetesPedigreeFunction': np.random.randint(0.078, 2.42, size=154),
    'Age': np.random.randint(21, 81, size=154)
    }

#converting the dictionar into df
unknown_df = pd.DataFrame(unknown_data)
unknown_df.describe()


# In[42]:


#scaling the data according to similar scaling done in main file 

# Use the scaler to transform new data
new_scaled_data = scaler.transform(unknown_df)

# Make predictions using the logistic regression model
predictions = lr.predict(new_scaled_data)



# In[43]:


#opening the Y_test cases for evaluation of model
with open('y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)


# In[44]:


# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


# In[ ]:




