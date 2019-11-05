# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:00:26 2019

@author: Adarsh
"""

import pandas as pd
import matplotlib.pylab as plt 
crime = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/clustering/crime_data.csv")

crime.columns
crime.isna().sum()
crime.isnull().sum()


# Normalization function suing z std. all are continuous data , not considering city variable.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
	  
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

# getting aggregate mean of each cluster
crime.iloc[:,2:].groupby(crime.clust).median()

# creating a csv file 
crime.to_csv("crime.csv",encoding="utf-8")

####succesfully formed 4 clusters, we can say that clusteer 2 has more no of rape, where urban population is more and murder is more for cluster 0.



##################################*********#######################################



####### trying with different linkage method ###########

#Continew here after doing normalization.

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

z = linkage(df_norm, method="average",metric="chebyshev")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=5,	linkage='average',affinity = "manhattan").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

# getting aggregate mean of each cluster
crime.iloc[:,2:].groupby(crime.clust).median()

# creating a csv file 
crime.to_csv("crime.csv",encoding="utf-8")

import os
os.getcwdb()#to get current working directory

###########################################################****###############################################
#crime dataset is very low so its better to perform hierarchical clustering.
#kmenas clustering suitable for larger datasets, so no need to  perform kmeans as we got clusters properly using Hierarchical.