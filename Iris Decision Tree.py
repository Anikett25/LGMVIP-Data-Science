#!/usr/bin/env python
# coding: utf-8

# # LGMVIP Data Analytics
# 
# ## Task 2- Prediction using Decision Tree algorithm
# 
# ### Name: Aniket Taru

# In[1]:


#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#loading the dataset
iris = load_iris()
x = iris.data[:,:]
y = iris.target


# In[3]:


#Input data
df = pd.DataFrame(iris['data'],columns=['Petal length','Petal width','Sepal length','Sepal width'])
df['Species'] = iris['target']
df['Species'] = df['Species'].apply(lambda x: iris['target_names'][x])

df.head()


# In[4]:


#Size of the dataset
df.shape


# In[5]:


df.describe()


# In[6]:


#EDA
sns.pairplot(df)


# In[7]:


#Scatter plot based on the features of species
sns.FacetGrid(df, hue ='Species').map(plt.scatter,'Petal length','Petal width').add_legend()
plt.show()

sns.FacetGrid(df, hue ='Species').map(plt.scatter,'Sepal length','Sepal width').add_legend()
plt.show()


# In[8]:


#Data Preprocessing
x= df.iloc[:,[0,1,2,3]]
y= df.Species


# In[9]:


#Splitting the data in training and testing set
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)
print("Training complete")


# In[10]:


y_pred_train = classifier.predict(x_train)
y_pred = classifier.predict(x_test)


# **Accuracy of the model**

# In[11]:


#For training set
accuracy_score(y_train,y_pred_train)


# In[12]:


#For Testing set
accuracy_score(y_test, y_pred)


# In[13]:


#Comparing the actual and predicted classification
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result.tail()


# **Visualizing the decision tree**

# In[14]:


rcParams['figure.figsize'] = 70,40
plot_tree(classifier, precision=5, rounded=True, filled=True)


# In[ ]:




