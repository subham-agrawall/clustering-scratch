import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/PA1.txt', sep="	", header=None)
df.columns = ["x1", "x2", "cluster"]
print df["cluster"].value_counts()

df.plot(kind="scatter", x="x1",y="x2")
plt.show()

pts = df.drop("cluster", 1)
points = pts.as_matrix()

def initial_centroids(points, k):
	centroids = points.copy()
	np.random.shuffle(centroids)
	return centroids[:k]

def get_labels(points, centroids):
	distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
	return np.argmin(distances, axis=0)

def move_centroids(points, closer, centroids):
	return np.array([points[closer==k].mean(axis=0) for k in range(centroids.shape[0])])

def has_converged(oldCentroids, centroids, iterations):
	Max_iterations = 1000
	if iterations>Max_iterations:
		return True
	return np.all(oldCentroids==centroids)

def kmeans(dataSet, k):
	#initialize centroids randomly
	centroids=initial_centroids(dataSet, k)
	iterations=0
	oldCentroids=None
	while not (has_converged(oldCentroids, centroids, iterations)):
		#For convergence test
		oldCentroids=centroids
		iterations+=1
		labels = get_labels(dataSet, centroids)
		centroids = move_centroids(dataSet, labels, centroids)
	return (centroids, labels)

def perf_measure(y_actual, y_hat,k):
	TP = 0
	for i in range(len(y_hat)): 
		if y_actual[i]==y_hat[i]==k:
			TP += 1
	return(TP)

centroid, label = kmeans(points,2)

import matplotlib
colors=['red','green']
plt.scatter(points[:,0], points[:,1], c=label, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

label = [x+1 for x in label]
# number of correctly assigned cluster points  
clus1= perf_measure(df["cluster"],label,1)
clus2= perf_measure(df["cluster"],label,2)
# true positive rate
tpr_clus1 = 100*clus1/202
tpr_clus2 = 100*clus2/199
print tpr_clus1, tpr_clus2 