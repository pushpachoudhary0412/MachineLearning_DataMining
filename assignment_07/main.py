from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=250, centers=3, n_features=2, random_state=42)

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

print('Labels sample:', labels[:15])
print('Silhouette Score:', round(silhouette_score(X, labels), 3))
