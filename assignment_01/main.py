import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)

# Sample dataset
df = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 35, np.nan, 50],
    'salary': [50000, 60000, 55000, np.nan, 65000, 70000, 80000],
    'city': ['Berlin', 'Munich', 'Berlin', np.nan, 'Hamburg', 'Munich', 'Berlin'],
    'purchased': [0, 1, 0, 1, 1, 0, 1]
})

X = df.drop('purchased', axis=1)
y = df['purchased']

num_features = ['age', 'salary']
cat_features = ['city']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print('Original shape:', X.shape)
print('Processed train shape:', X_train_processed.shape)
print('Processed test shape:', X_test_processed.shape)
print('Preprocessing completed successfully.')
