# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:05:58 2019

@author: Adarsh
"""

import pandas as pd
import matplotlib.pylab as plt 
airline = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/clustering/EastWestAirlines.xlsx",sheet_name="data")

airline.columns

# Droping first column 
airline.drop(["ID#"],inplace=True,axis = 1)

# Normalization function 
def norm_func(i):
	x = (i-i.min())	/	(i.max()	-	i.min())
	return(x)
	  
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline.iloc[:,:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


###################################

#By seeing dendrogram plot we cant decide no. of clusters because the dataset is very large and Heirarchical clustering is suitable for small datasets.
# so we will perform using in Kmeans,


#############################################*****#######################################################

#Non Hierarchical clustering,

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

airline = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/clustering/EastWestAirlines.xlsx",sheet_name="data")

airline.columns

# Droping first column 
airline.drop(["ID#"],inplace=True,axis = 1)

# Normalization function 
def norm_func(i):
	x = (i-i.min())	/	(i.max()	-	i.min())
	return(x)
	  
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline.iloc[:,:])
df_norm.describe()

###### screw plot or elbow curve ############
k = (range(2,11))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airline['clust']=md # creating a  new column and assigning it to new column 

airln = airline.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]

x=airln.iloc[:,1:].groupby(airln.clust).mean()

airln.to_csv("airline.csv")
import os
os.getcwdb()#to get current working directory

#for this dataset 5 clusters is good to name the clusters and the TWSSD is less for 5 clusters.
