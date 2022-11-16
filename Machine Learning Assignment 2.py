#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv("Data1.csv")
df = (df - np.mean(df, axis = 0)) / np.std(df)
x = df.drop(df.columns[len(df.columns)-1], axis=1)
y = df.iloc[:, -1]


# In[3]:


#Covariance Matrix
cov = np.cov(x,  rowvar = False)
cov = pd.DataFrame(cov)
u, s, vh = np.linalg.svd(cov)


# In[4]:


#eigen vectors and values found
eigen_vec = u.T
eigen_vals = s

#explained variance 
var1 = eigen_vals[0] / np.sum(eigen_vals)
var2 = eigen_vals[1] / np.sum(eigen_vals)
var3 = eigen_vals[2] / np.sum(eigen_vals)
var4 = eigen_vals[3] / np.sum(eigen_vals)
variance = [var1, var2, var3, var4]


# In[5]:


#Loadings/Coeffiecients for the Principal Components and Component Matrix
loadings = np.dot(x, eigen_vec * np.sqrt(eigen_vals))
loadings = pd.DataFrame(loadings, columns = ['PC1', 'PC2', 'PC3', 'PC4'])


# In[6]:


#PC scores and Correlation Matrix
pc_scores = np.dot(x, eigen_vec)
x_corr = x.corr().round(3)
sns.heatmap(x_corr, annot = True, vmax = 1, vmin = -1, center = 0)


# In[7]:


#Projection Matrix
proj_matrix = np.dot(x, eigen_vec)
proj = pd.DataFrame(proj_matrix)


# In[8]:


#Kaiser Criteria
avg_eig_val = np.mean(eigen_vals)
kaiser = eigen_vals[eigen_vals > avg_eig_val]
df_kaiser = pd.DataFrame(kaiser).T
df_kaiser.columns = ['PC1', 'PC2']


# In[9]:


sns.pairplot(data = df, x_vars = ['T', 'P', 'TC', 'SV'], y_vars = 'Idx', height = 7, aspect = 0.7)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(loadings, y, test_size = 0.25, random_state = 0)

linmodel = LinearRegression()
linmodel = linmodel.fit(x_train, y_train)


# In[11]:


linemodel_pred = linmodel.predict(x_train)
linemodel_test_pred = linmodel.predict(x_test)


# In[12]:


print('RMSE Train (Non-PCA):', np.sqrt(mean_squared_error(y_train, linemodel_pred)))
print('R^2 Train (Non-PCA)', r2_score(y_train, linemodel_pred))
print('RMSE Test (Non-PCA):', np.sqrt(mean_squared_error(y_test, linemodel_test_pred)))
print('R^2 Test (Non-PCA)', r2_score(y_test, linemodel_test_pred))


# In[13]:


loadings = pd.DataFrame(loadings)
PCA_df = loadings.iloc[:, [0,1]]


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(PCA_df, y, test_size = 0.25, random_state = 0)

PCA_linmodel = LinearRegression()
PCA_linmodel = PCA_linmodel.fit(x_train, y_train)


# In[15]:


PCA_linemodel_pred = PCA_linmodel.predict(x_train)
PCA_linemodel_test_pred = PCA_linmodel.predict(x_test)


# In[16]:


print('RMSE Train (PCA):', np.sqrt(mean_squared_error(y_train, PCA_linemodel_pred)))
print('R^2 Train (PCA)', r2_score(y_train, PCA_linemodel_pred))
print('RMSE Test (PCA):', np.sqrt(mean_squared_error(y_test, PCA_linemodel_test_pred)))
print('R^2 Test (PCA)', r2_score(y_test, PCA_linemodel_test_pred))

