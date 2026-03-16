from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# DBSCAN may mark some points as noise (-1). Compute silhouette only on non-noise points.
mask = dbscan_labels != -1
if mask.sum() > 1 and len(set(dbscan_labels[mask])) > 1:
    dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
else:
    dbscan_silhouette = None

print('K-Means Cluster Centers:\n', kmeans.cluster_centers_)
print('K-Means Inertia:', round(kmeans.inertia_, 3))
print('K-Means Silhouette Score:', round(kmeans_silhouette, 3))

noise_points = int((dbscan_labels == -1).sum())
clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print('DBSCAN Clusters Found:', clusters_found)
print('DBSCAN Noise Points:', noise_points)
if dbscan_silhouette is not None:
    print('DBSCAN Silhouette Score:', round(dbscan_silhouette, 3))
else:
    print('DBSCAN Silhouette Score: Not defined (insufficient non-noise clusters)')
