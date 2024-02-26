#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the CSV file and convertin it into a Data Frame
df = pd.read_csv("diabetes.csv")


# In[3]:


#Printing the top 4 columns of the dataframe
df.head(4)


# In[5]:


# Check for null values in the DataFrame
null_values = df.isnull().sum()

# Display the count of null values for each column
print(null_values)


# In[6]:


# When you call df.describe(), it computes summary statistics of the numerical columns in the DataFrame.
#These statistics include count, mean, standard deviation, minimum, maximum, and various quantiles 
#(25th, 50th, and 75th percentiles).

df.describe()


# Some of the data is Zero, which is wrong, such as Glucose, BloodPressure, Insulin, BMI, SO correcting these columns

# In[8]:


# we are going to replace these zeros with their mean value"

df['BMI']=df['BMI'].replace(0, df['BMI'].mean())
df['Insulin']=df['Insulin'].replace(0, df['Insulin'].mean())
df['Glucose']=df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0, df['SkinThickness'].mean())


# In[9]:


df.describe()


# In[44]:


# Set Seaborn style
sns.set_style('whitegrid')

# Plot density plot for all features
df.hist(bins=50, figsize=(20, 15))
plt.show()


# In[41]:


columns_to_include = df.drop(columns=['Outcome']) 

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(columns_to_include.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Between Columns (Excluding "exclude_column")')
plt.show()


# In[10]:


fig, ax =plt.subplots(figsize=(15,10))
sns.boxplot(data=df, width=0.5, ax=ax, fliersize=3)


# In[13]:


#Splitting the data into X(Input Variable) and y(Output Variable)
X=df.drop(columns= ['Outcome'])
y=df['Outcome']


# In[14]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


np.size(y_test)


# In[15]:


# Initialize the logistic regression model
lr = LogisticRegression()


# In[19]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


# Fit the model on the training data
lr.fit(X_train_scaled, y_train)


# In[22]:


# Make predictions on the testing data
predictions = lr.predict(X_test_scaled)


# In[31]:


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


# **Interpretation of the values**
# 
# __Accurcay__ :- An accuracy of 0.766 indicates that approximately 76.62% of the predictions made by the model are correct.
# 
# __Precision__:- A precision of 0.686 indicates that approximately 68.63% of the samples predicted as positive by the model are actually positive.
# 
# __Recall__:- A recall of 0.636 indicates that approximately 63.64% of the actual positive samples in the dataset are correctly identified by the model.
# 
# __F1__:- An F1 Score of 0.660 indicates that the model achieves a good balance between precision and recall.
# 
# **ROC AUC Curve**:- A ROC AUC score of 0.737 indicates that the model has reasonable discriminative power in distinguishing between positive and negative classes.

# In[38]:


# Plot ROC curve
y_prob = lr.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[39]:


coefficients = lr.coef_
intercept = lr.intercept_

print("Intercept:", intercept)
print("Coefficients:", coefficients)


# # Impact of Itercept and Coefficient 
# 
# __Intercept__ is -0.8738, so when all the predictor variables are set to zero.Since the log odds are negative, it suggests that the baseline probability of diabetes when all predictors are zero is less than 0.5 
# 
# __Pregnancies__: A one-unit increase in the number of pregnancies is associated with a 0.222 increase in the log odds of diabetes.
# 
# __Glucose__: A one-unit increase in glucose level is associated with a 1.125 increase in the log odds of diabetes.
# 
# __BloodPressure__: A one-unit increase in blood pressure is associated with a -0.168 decrease in the log odds of diabetes.
# 
# __SkinThickness__: A one-unit increase in skin thickness is associated with a 0.016 increase in the log odds of diabetes.
# 
# __Insulin__: A one-unit increase in insulin level is associated with a -0.186 decrease in the log odds of diabetes.
# 
# __BMI__: A one-unit increase in BMI is associated with a 0.731 increase in the log odds of diabetes.
# 
# __DiabetesPedigreeFunction__: A one-unit increase in the diabetes pedigree function value is associated with a 0.212 increase in the log odds of diabetes.
# 
# __Age__: A one-unit increase in age is associated with a 0.393 increase in the log odds of diabetes.
# 
# 
# 

# In[57]:


# Save the logistic regression model to a pickle file
imort pickle
    
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump((scaler, lr), file)    


# In[58]:


# making the data availbel in pickle file to test against the test data 
with open('y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)


# In[ ]:


"Tried LAsso, ridge and Elastic net, but these model pridict the same result. So not included in here"

