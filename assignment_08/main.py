from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print('Original shape:', X.shape)
print('Reduced shape:', X_pca.shape)
print('Explained variance ratio (first 10):')
print(pca.explained_variance_ratio_)
print('Cumulative explained variance:', round(pca.explained_variance_ratio_.sum(), 4))
