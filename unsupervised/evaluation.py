
# clustering

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score

data=pd.read_csv("Mall_Customers.csv")
X=data[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans=KMeans(n_clusters=5,random_state=42)
labels=kmeans.fit_predict(X)

print("Silhouette Score:", silhouette_score(X, labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X, labels))
print("Inertia (SSE):", kmeans.inertia_)





#  dim Reduction ------>  PCA variance ratio, Visualization (t-SNE, PCA scatter)
#        |
#        |
#        |
#        V

# Reduce features → improve classification/clustering later(3 data to 2 data)