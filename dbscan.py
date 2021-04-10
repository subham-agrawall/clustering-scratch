import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unionfind import *

df = pd.read_csv('./data/PA1.txt', sep="	", header=None)
df.columns = ["x1", "x2", "cluster"]
df.head()
print df["cluster"].value_counts()
pts = df.drop("cluster", 1)
points = pts.as_matrix()

k=4 # its a reasonable value for most two-dimensional data sets

def proximity(dataSet):
	# Proximity matrix for Euclidean distance
	proximity = np.sqrt(((dataSet - dataSet[:, np.newaxis])**2).sum(axis=2))
	# Distance of points in ascending order (in row)
	dist = np.sort(proximity, axis=1)
	# Index of points with increasing distance for every point (in row)
	index = np.argsort(proximity, axis=1)
	return dist, index
	
# k-dist
def k_dist(dist, k):
	kth_nearest_neighbor = dist[:,k]
	kth_asc = np.sort(kth_nearest_neighbor)
	return kth_asc

# Determining epsilon from the graph
def epsilon(kth_asc):
	slope = kth_asc[1:] - kth_asc[:-1]
	i = 0
	while(slope[i]<0.05):
		i += 1    
	eps = kth_asc[i]
	return eps

# label points
def label_points(dataSet, k, epsilon, dist, index):
	label = []
	core = []
	border = []
	for i in range(len(dataSet)):
		if dist[i,k-1]<epsilon:
			label.append('core')
			core.append(i)
		else: 
			label.append(None)
	for	i in range(len(dataSet)):
		if ((label[i]==None) and ((label[index[i,1]]=='core' and dist[i,1]<epsilon) or (label[index[i,2]]=='core' and dist[i,2]<epsilon))):
			label[i]='border'
			border.append(i)
		elif label[i]==None:
			label[i]='noise'
	return label, core, border

# list of core points
def core_matrix(points, core):
	core_pts=np.random.randn(len(core),2)
	for i in range(len(core)):
		core_pts[i,:]=points[core[i],:]
	return core_pts

# The algorithm designed by Rosenfeld and Pfaltz in 1966 which uses union-find data structure 
def first_pass(dist_core, index_core, epsilon):
	core_label= [None]*len(dist_core)
	labelCount=0
	uf=UnionFind(1000)
	for i in range(len(dist_core)):
		j=0
		list=[]
		while(dist_core[i,j]<epsilon):
			point=index_core[i,j]
			a=core_label[point]
			if (a ==None):
				j+=1
			else:
				list.append(a)
				j+=1
		if len(list)==0:
			core_label[i]=labelCount
			labelCount += 1
		else:
			val=min(list)
			core_label[i]=val
			for k in range(len(list)):
				uf.union(list[k],val)			
	return core_label, uf
	
def merge_labels(first_label, uf):
	for i in range(len(first_label)):
		a=uf.find(first_label[i])
		first_label[i]=a
	return first_label

def perf_measure(y_actual, y_hat,k):
	TP = 0
	for i in range(len(y_hat)): 
		if y_actual[i]==y_hat[i]==k:
			TP += 1
	return(TP)

# k-dist PLOT
dist, index = proximity(points)
fourth_asc = k_dist(dist, k)
plt.plot(np.arange(431), fourth_asc)
plt.xlim(-10,440)
plt.ylim(0,6)
plt.xlabel(r'Points Sorted By Distance To 3rd Nearest Neighbor')
plt.ylabel(r'4th Nearest Neighbor Distance')
plt.grid()
plt.show()
eps = epsilon(fourth_asc)
print eps

label, core, border = label_points(points, 4, eps, dist, index)
core_pts = core_matrix(points,core)
dist_core, index_core = proximity(core_pts)
core_label, uf = first_pass(dist_core,index_core,eps)
core_label = merge_labels(core_label, uf) 
core_label = [1 if x==2 else 2 for x in core_label]
cluster = [0]*len(points)
for x in range(len(core_label)):
	cluster[core[x]]=core_label[x]
# assigning border points
for x in border:
	cluster[x]=cluster[index[x,1]] 

import matplotlib
colors=['red','green','blue']
plt.scatter(points[:,0], points[:,1], c=cluster, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

# number of correctly assigned cluster points  
clus1= perf_measure(df["cluster"],cluster,1)
clus2= perf_measure(df["cluster"],cluster,2)
# true positive rate
tpr_clus1 = 100*clus1/202
tpr_clus2 = 100*clus2/199
print tpr_clus1, tpr_clus2

# Plot of the clusters given to us
colors=['red','green','blue']
plt.scatter(points[:,0], points[:,1], c=df['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
plt.show()