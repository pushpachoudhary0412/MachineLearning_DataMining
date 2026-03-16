from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)

print('Decision Tree Accuracy:', round(accuracy_score(y_test, pred_dt), 4))
print('Random Forest Accuracy:', round(accuracy_score(y_test, pred_rf), 4))
