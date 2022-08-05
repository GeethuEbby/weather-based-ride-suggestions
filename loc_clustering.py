#!/usr/bin/env python
# coding: utf-8

# The function is used to identify clusters of locations in the dataset. The dataframe with the location details is passed as an input to the function, along with the column range of the dataframe to be considered and the number of clusters. KMeans clustering method will compute the clusters and will add a column called 'cluster_label' to the data frame. The dataframe with cluster labels is returned as the output.

# In[1]:


# Importing libraries
import pandas as pd
from sklearn.cluster import KMeans


# In[7]:


def cluster_fn(df,col_beg,col_end,score):
    # Clustering using K = 7 and assigning Clusters to the dataset
    kmeans = KMeans(n_clusters = score, init ='k-means++')
    kmeans.fit(df[df.columns[col_beg:col_end]]) # Compute k-means clustering.
    df['cluster_label'] = kmeans.fit_predict(df[df.columns[col_beg:col_end]])
    return df


# In[ ]:




