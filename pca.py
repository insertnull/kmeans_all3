import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances
from sklearn.decomposition import PCA

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Define genres
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Assign binary genre flags
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with genre information
df = ratings.merge(movies, on='movieId')

# Compute user average rating per genre
user_genre_ratings = df.groupby('userId')[genres].mean()
user_genre_ratings_original = user_genre_ratings.copy()

# Standardize original data
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS CLUSTERING --- ###
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings_original["Cluster_Original"] = clusters_original

### --- OUTLIER REMOVAL USING LOF --- ###
lof = LocalOutlierFactor(n_neighbors=20, metric="euclidean", contamination=0.1)
lof_scores = -lof.fit_predict(user_genre_ratings_scaled)
lof_values = -lof.negative_outlier_factor_

threshold = 1.5
mask = lof_values < threshold
if np.sum(mask) < 30:
    print("Too few inliers after LOF, skipping outlier removal.")
    user_genre_ratings_filtered = user_genre_ratings.copy()
    user_genre_ratings_filtered_scaled = user_genre_ratings_scaled
else:
    user_genre_ratings_filtered = user_genre_ratings[mask]
    user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

### --- CUSTOM K-MEANS USING CANBERRA + OPTIMAL k (CH Index) --- ###
def custom_kmeans_canberra(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric='canberra')
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(centers[i])
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers

# Try different k using Calinski-Harabasz index
best_k = None
best_score = -np.inf
best_labels = None

for k in range(2, 16):
    labels, _ = custom_kmeans_canberra(user_genre_ratings_filtered_scaled, k)
    score = calinski_harabasz_score(user_genre_ratings_filtered_scaled, labels)
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

clusters_enhanced = best_labels
user_genre_ratings_filtered["Cluster_Enhanced"] = clusters_enhanced

### --- UPDATED PCA VISUALIZATION WITH GENRE INFLUENCE --- ###
def pca_visualization_with_genres(data_scaled, labels, title, feature_names):
    # Fit PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)

    # Get genre contributions to PC1 and PC2
    components = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2'])

    # Get top 3 contributing genres (by absolute value) for each principal component
    top_pc1 = components.loc['PC1'].abs().sort_values(ascending=False).head(5).index.tolist()
    top_pc2 = components.loc['PC2'].abs().sort_values(ascending=False).head(5).index.tolist()

    # Make custom axis labels using top contributing genres
    xlabel = f"{', '.join(top_pc1)}"
    ylabel = f"{', '.join(top_pc2)}"

    # Plot with PCA
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(labels):
        plt.scatter(reduced[labels == cluster, 0],
                    reduced[labels == cluster, 1],
                    label=f'Cluster {cluster}',
                    alpha=0.7)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print full component loadings
    print("\nGenre Contributions to PCA Components:")
    print(components.T.sort_values(by="PC1", ascending=False))

# --- Run updated PCA visualization ---
pca_visualization_with_genres(user_genre_ratings_scaled, clusters_original,
                               "Original K-Means (Euclidean)", genres)

pca_visualization_with_genres(user_genre_ratings_filtered_scaled, clusters_enhanced,
                               "Enhanced K-Means (Canberra + LOF + CH Index)", genres)

### --- SILHOUETTE SCORES --- ###
sil_original = silhouette_score(user_genre_ratings_scaled, clusters_original)
sil_enhanced = silhouette_score(user_genre_ratings_filtered_scaled, clusters_enhanced)

print("\nSilhouette Score (All 16 Genres):")
print(f"  Original K-Means (Euclidean): {sil_original:.4f}")
print(f"  Enhanced K-Means (Canberra + LOF + CH Index): {sil_enhanced:.4f}")

### --- SUMMARY --- ###
print("\nUser Clustering Summary:")
print("  Total users:", user_genre_ratings.shape[0])
print("  Inliers after LOF filtering:", user_genre_ratings_filtered.shape[0])
print("  Outliers removed:", user_genre_ratings.shape[0] - user_genre_ratings_filtered.shape[0])
print("  Optimal k for enhanced clustering (CH Index):", best_k)
