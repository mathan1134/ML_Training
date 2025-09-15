#                     kmeans

# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)

# kmeans=KMeans(n_clusters=4,random_state=42)
# y_pred=kmeans.fit_predict(X)

# print("op :",y_pred)

# plt.scatter(X[:,0],X[:,1],c=y_pred,s=50,cmap="viridis")
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="red",marker="X",s=200,label="centroids")
# plt.legend()
# plt.title("kmeans")
# plt.show()

# #                hierachical 

# from sklearn.cluster import AgglomerativeClustering

# X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.8,random_state=42)

# he=AgglomerativeClustering(n_clusters=3)
# h_pred=he.fit_predict(X)

# print("hr :",h_pred)

# plt.scatter(X[:,0],X[:,1],c=h_pred,cmap="rainbow")
# plt.legend()
# plt.title("hirechical")
# plt.show()

# #             DBSCAN

# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_moons

# X,_=make_moons(n_samples=300,noise=0.05,random_state=42)

# dn=DBSCAN(eps=0.2,min_samples=5)
# y_pred=dn.fit_predict(X)
# print("DBSCAN :",y_pred)

# plt.scatter(X[:,0],X[:,1],c=y_pred,cmap="plasma")
# plt.legend()
# plt.title("DBScan")
# plt.show()


#       with real data


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("Mall_Customers.csv")
X=data[['Annual Income (k$)',"Spending Score (1-100)"]]

# kmeans=KMeans(n_clusters=7,random_state=42)
# labels=kmeans.fit_predict(X)

# data["Cluster"]=labels                                         

# plt.scatter(X["Annual Income (k$)"],X["Spending Score (1-100)"],c=labels,cmap="viridis",s=50)
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c="red",marker="X",s=200,label="centroids")
# plt.xlabel("annual income")
# plt.ylabel("spending score")
# plt.legend()
# plt.title("kmeans")
# plt.show()

# for i in range(7):
#     print(f"\n--- Cluster {i} ---")
#     print(data[data["Cluster"] == i])


X_scaled=StandardScaler().fit_transform(X)

db=DBSCAN(eps=0.2,min_samples=8)
labels = db.fit_predict(X_scaled)

data["Cluster"] = labels

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"],                       # not good for these
            c=labels, cmap="plasma", s=50)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering")
plt.show()

for cluster in sorted(data["Cluster"].unique()):
    print(f"\n--- Cluster {cluster} ---")
    print(data[data["Cluster"] == cluster])

