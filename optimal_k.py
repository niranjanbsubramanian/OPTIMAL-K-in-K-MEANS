import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=43)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# homogeneity, completeness, and v-measure
k = [2, 3, 4, 5, 6, 7, 8]

homo_score = []
comp_score = []
vm_score = []

for n_cluster in k:
    y_pred = KMeans(n_clusters = n_cluster, max_iter=1000, random_state=47).fit_predict(X)
    homo = metrics.homogeneity_score(y, y_pred)
    comp = metrics.completeness_score(y, y_pred)
    vm = metrics.v_measure_score(y, y_pred)

    homo_score.append(homo)
    comp_score.append(comp)
    vm_score.append(vm)
plt.plot(k, homo_score, 'r', label='Homogeneity') 
plt.plot(k, comp_score, 'b', label='Completeness') 
plt.plot(k, vm_score, 'y', label='V- Measure')
plt.xlabel('Value of K')
plt.ylabel('homogeneity_completeness_v_measure')
plt.legend(loc=4) 
plt.show()

# Adjusted Rand Index
k = [2, 3, 4, 5, 6, 7, 8]
scores = []

for n_cluster in k:
    y_pred = KMeans(n_clusters = n_cluster, max_iter=1000, random_state=47).fit_predict(X)
    score = metrics.adjusted_rand_score(y, y_pred)
    scores.append(score)

plt.plot(k, scores, 'o-')
plt.title('Adjusted Rand Index')
plt.show()

# Elbow Method
k = [2, 3, 4, 5, 6, 7, 8]
inertias = []
for i in k:
    km = KMeans(n_clusters=i, max_iter=1000, random_state=47)
    km.fit(X)
    inertias.append(km.inertia_)
plt.plot(k, inertias, 'o-')
plt.xlabel("Value of k")
plt.ylabel("WSS")
plt.title('Elbow Method')
plt.show()

# Silhouette Score
k = [2, 3, 4, 5, 6, 7, 8]
score=[]
for n_cluster in k:
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    score.append(metrics.silhouette_score(X,kmeans.labels_))

plt.plot(k, score, 'o-')
plt.xlabel("Value for k")
plt.ylabel("Silhouette score")
plt.title('Silhouette Method')
plt.show()

# DB-Index
k = [2, 3, 4, 5, 6, 7, 8]
scores = []

for i in k:
    y_pred = KMeans(n_clusters = i, max_iter=1000, random_state = 43).fit_predict(X)
    score = metrics.davies_bouldin_score(X, y_pred)
    scores.append(score)
    print(score)

plt.plot(k, scores, 'o-')
plt.title('DAVIES-BOULDIN')
plt.show()

# CH-Index
k = [2, 3, 4, 5, 6, 7, 8]
scores = []

for i in k:
    y_pred = KMeans(n_clusters = i, max_iter=1000, random_state = 43).fit_predict(X)
    score = metrics.calinski_harabaz_score(X, y_pred)
    scores.append(score)
    print(score)

plt.plot(k, scores, 'o-')
plt.title('CALINSKI-HARABASZ')
plt.show()

# Gap Statistic
# https://github.com/milesgranger/gap_statistic/blob/master/Example.ipynb
from gap_statistic import OptimalK
optimalK = OptimalK(parallel_backend='None')
n_clusters = optimalK(X, cluster_array=np.arange(1, 10))

plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value)

plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()
