from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from gsrfs import GSRSelector

X, _ = load_wine(return_X_y=True)
pipe = Pipeline([
    ("select", GSRSelector(n_features=5, n_permutations=20, random_state=42)),
    ("scale", StandardScaler()),
    ("cluster", KMeans(n_clusters=3, n_init=20, random_state=42)),
])
pipe.fit(X)
print(pipe.named_steps["select"].get_support(indices=True))
