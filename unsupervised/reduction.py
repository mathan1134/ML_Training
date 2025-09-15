# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# # Create simple customer data
# data = pd.DataFrame({
#     "Age": [20, 25, 30, 35, 40, 45, 50],
#     "Annual Income (k$)": [15, 25, 35, 45, 55, 65, 75],
#     "Spending Score (1-100)": [80, 70, 65, 60, 40, 30, 20]
# })

# print(data)

# # Plot Age vs Income
# plt.scatter(data["Age"], data["Annual Income (k$)"], c="blue", s=50)
# plt.title("Age vs Income")
# plt.xlabel("Age")
# plt.ylabel("Annual Income (k$)")
# plt.show()

# Plot Income vs Spending
# plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"], c="green", s=50)
# plt.title("Income vs Spending")
# plt.xlabel("Annual Income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.show()




# X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
# X_scaled = StandardScaler().fit_transform(X)


                                    #    convert 3d data to 2d


# # Apply PCA to reduce 3 features → 2
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Put PCA results into a DataFrame
# pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# plt.scatter(pca_df["PC1"], pca_df["PC2"], c="red", s=100)
# plt.title("PCA Projection (2D)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()




#                                  # ---      t-SNE Example (reduce to 2D) ---   but visulaization
# tsne = TSNE(n_components=2, perplexity=3, random_state=42)
# X_tsne = tsne.fit_transform(X_scaled)

# plt.scatter(X_tsne[:,0], X_tsne[:,1], c='green', s=50)
# plt.title("t-SNE (2D Projection for visualization)")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans

# data = pd.read_csv("Mall_Customers.csv")

# X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
# X_scaled = StandardScaler().fit_transform(X)

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_tsne = tsne.fit_transform(X_scaled)

# kmeans = KMeans(n_clusters=5, random_state=42)
# labels = kmeans.fit_predict(X_scaled)

# data["Cluster"] = labels
# data["TSNE-1"] = X_tsne[:,0]
# data["TSNE-2"] = X_tsne[:,1]

# plt.scatter(data["TSNE-1"], data["TSNE-2"], c=labels, cmap="viridis", s=50)
# plt.title("t-SNE + KMeans Clusters")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.show()

# for i in range(5):
#     print(f"\n--- Cluster {i} ---")
#     print(data[data["Cluster"] == i][["CustomerID","Age","Annual Income (k$)","Spending Score (1-100)"]])
