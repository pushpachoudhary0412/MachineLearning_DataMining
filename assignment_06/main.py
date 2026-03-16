from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

sil_score = silhouette_score(X, labels)

print('Cluster Centers:
', kmeans.cluster_centers_)
print('Inertia:', round(kmeans.inertia_, 3))
print('Silhouette Score:', round(sil_score, 3))
