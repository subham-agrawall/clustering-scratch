# Clustering from scratch
This repo contains implementation of k-means and DBSCAN algorithm from scratch on a sample dataset. 

## Dataset
In the below figure, green and blue points represent cluster 1 and cluster 2 respectively. Red points represent noise.  
<img src="https://github.com/subham-agrawall/clustering-scratch/blob/main/output/dataset.jpeg" width="400" height="300">

## K-Means output
Applying K-means clustering algorithm for given dataset with k=2,  
<img src="https://github.com/subham-agrawall/clustering-scratch/blob/main/output/kmeans.jpeg" width="400" height="300">  
TRUE POSITIVE RATE FOR CLUSTER-1 =  15%  
TRUE POSITIVE RATE FOR CLUSTER-2 =  16%  
No noise points

## DBSCAN output
<img src="https://github.com/subham-agrawall/clustering-scratch/blob/main/output/kdist.jpeg" width="350" height="250">
As observed from the above figure and also from code, we get epsilon=1.22 for given data and k=4.  


Applying DBSCAN algorithm with a value of k=4,  
<img src="https://github.com/subham-agrawall/clustering-scratch/blob/main/output/dbscan.jpeg" width="400" height="300">  
TRUE POSITIVE RATE FOR CLUSTER-1 =  100%  
TRUE POSITIVE RATE FOR CLUSTER-2 =  100%  
