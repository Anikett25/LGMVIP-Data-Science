#!/usr/bin/env python
# coding: utf-8

# # LGMVIP Data Analytics
# 
# ## Task 1 - Classification using Unsupervised ML algorithm
# 
# ### Name: Aniket Taru

# In[2]:


#Importing libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[3]:


#reading the file
data = pd.read_csv(r"D:\Iris.csv")
data.head()


# In[4]:


#Size of data
data.shape


# In[5]:


#Data description
data.info()


# In[6]:


#Dropping Id column
data.drop('Id', axis=1, inplace = True)
data.columns


# In[8]:


#checking for duplicate observations
print("\nNumber of duplicate observations\n",data.duplicated().sum())


# In[9]:


#dropping duplicate observations
data.drop_duplicates(inplace=True)


# In[12]:


#Checking for outliers
for i in data.columns:
    if data[i].dtype=='float64':
        plt.figure(figsize=(6,3))
        sns.boxplot(data[i])
        plt.show()


# In[14]:


#Distribution of Species
sns.countplot(data.Species)
print(data.Species.value_counts())


# In[15]:


data.describe()


# In[18]:


#Distribution of Features
for i in data.columns[:-1]:
    sns.kdeplot(data = data.loc[data.Species=='Iris-setosa'][i], label='Iris-setosa', shade=True)
    sns.kdeplot(data = data.loc[data.Species=='Iris-versicolor'][i], label='Iris-versicolor', shade=True)
    sns.kdeplot(data = data.loc[data.Species=='Iris-virginica'][i], label='Iris-virginica', shade=True)
    plt.title(i);
    plt.show()


# In[19]:


#Correlation matrix
data.corr()


# In[21]:


plt.figure(figsize=(8,4))
sns.heatmap(abs(data.corr()), cmap='Blues', annot=True)


# **K-means clustering To predict number of clusters**

# In[23]:


from sklearn.cluster import KMeans


# In[24]:


SSE=[]
for i in range(1,10):
    kmeans = KMeans(n_jobs=-1, n_clusters=i, init='k-means++')
    kmeans.fit(data.iloc[:,[0,1,2,3]])
    SSE.append(kmeans.inertia_)


# In[27]:


ctab = pd.DataFrame({"Cluster":range(1,10), 'SSE':SSE})
plt.figure(figsize=(10,5))
plt.plot(ctab["Cluster"], ctab["SSE"], marker='o')
plt.title("Elbow method for optimum K value")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()


# In[28]:


kmeans = KMeans(n_jobs=-1, n_clusters=3, init="k-means++")
kmeans.fit(data.iloc[:,[0,1,2,3]])
kmeans.cluster_centers_


# In[29]:


#Adding a new column to show the cluster
data['Cluster'] = kmeans.labels_
data


# In[30]:


#Total count of cluster and species
display(data['Cluster'].value_counts(), data['Species'].value_counts())


# In[31]:


#Plotting the centroids
plt.figure(figsize=(10,5))
plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'], c=data.Cluster)
plt.title("Predicted Clusters")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='blue', label='Centroids')
plt.show()


# In[34]:


#creating a new column to rectify the error generated in the model
data['Species_encoded'] = data['Species'].apply(lambda x: 1 if x=='Iris-setosa' else 2 if x=='Iris-virginica' else 0)
data


# **Model acurracy**

# In[35]:


print(classification_report(data['Species_encoded'], data['Cluster']))


# In[38]:


sns.heatmap(confusion_matrix(data['Species_encoded'], data['Cluster']), annot=True)

