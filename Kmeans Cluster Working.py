#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Kütüphaneleri import edelim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


# Create dataset-Datayı kendimiz oluşturalım
#ort 25 sigma 5 olan 1000 değer üretelim

#class1
x1=np.random.normal(25,5,1000)
y1=np.random.normal(25,5,1000)

#class2
x2=np.random.normal(55,5,1000)
y2=np.random.normal(60,5,1000)

#class3
x3=np.random.normal(55,5,1000)
y3=np.random.normal(15,5,1000)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)


dictionary={"x":x,"y":y}


# In[20]:


data=pd.DataFrame(dictionary)


# In[21]:


data.head()


# In[22]:


data.info()

#3000 değeri olan data oluşturmuş olduk


# In[23]:


#oluşturduğumuz dataları görselleştirelim

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()


# In[24]:


from sklearn.cluster import KMeans
wcss=[]


# In[25]:


#1 den 15 e kadar k değerlerini deneyelim
for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
# her bir k değeri için inertia değeri hesaplayacak


# In[26]:


plt.plot(range(1,15),wcss)
plt.xlabel("Number of cluster")
plt.ylabel("wcss")
plt.show()


# In[27]:


#Görselden de anlaşılacağı üzere en iyi k=3 değerinde modeli kurmuş oldu


# In[31]:


#k=3 için modeli deneyelim
kmeans2=KMeans(n_clusters=3)
clusters=kmeans2.fit_predict(data)
data["label"]=clusters
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
#küme merkezlerini de sarı ile görselleştirdik
plt.show()


# In[ ]:




