import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Extract genres
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Assign binary genre flags
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with movies
df = ratings.merge(movies, on='movieId')

# Compute user average ratings per genre
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# Make a copy before modifying
user_genre_ratings_original = user_genre_ratings.copy()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)

# Store clusters back in the original DataFrame
user_genre_ratings_original["Cluster_Original"] = clusters_original

### --- LOF OUTLIER REMOVAL ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers_lof = lof.fit_predict(user_genre_ratings_scaled)

# Keep only inliers
user_genre_ratings_filtered = user_genre_ratings[outliers_lof == 1]
user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

### --- ENHANCED K-MEANS (ONLY LOF FOR ENHANCEMENT) ---
kmeans_enhanced = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_enhanced = kmeans_enhanced.fit_predict(user_genre_ratings_filtered_scaled)

# Store clusters back in the filtered DataFrame
user_genre_ratings_filtered["Cluster_Enhanced"] = clusters_enhanced

### --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2)

# Apply PCA on the original data (before LOF removal)
pca_original = pca.fit_transform(user_genre_ratings_scaled)
user_genre_ratings_original["PCA1"] = pca_original[:, 0]
user_genre_ratings_original["PCA2"] = pca_original[:, 1]

# Apply PCA on the filtered data (after LOF removal)
pca_enhanced = pca.fit_transform(user_genre_ratings_filtered_scaled)
user_genre_ratings_filtered["PCA1"] = pca_enhanced[:, 0]
user_genre_ratings_filtered["PCA2"] = pca_enhanced[:, 1]

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS ---
def plot_side_by_side():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original K-Means Scatterplot
    ax = axes[0]
    for cluster in sorted(user_genre_ratings_original["Cluster_Original"].unique()):
        subset = user_genre_ratings_original[user_genre_ratings_original["Cluster_Original"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Original K-Means - PCA')
    ax.legend()

    # Enhanced K-Means (with LOF) Scatterplot
    ax = axes[1]
    for cluster in sorted(user_genre_ratings_filtered["Cluster_Enhanced"].unique()):
        subset = user_genre_ratings_filtered[user_genre_ratings_filtered["Cluster_Enhanced"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Enhanced K-Means (LOF) - PCA')
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_side_by_side()

### --- SILHOUETTE SCORES ---
# Compute silhouette scores for the original and enhanced K-Means
silhouette_original = silhouette_score(user_genre_ratings_scaled, user_genre_ratings_original["Cluster_Original"])
silhouette_enhanced = silhouette_score(user_genre_ratings_filtered_scaled, user_genre_ratings_filtered["Cluster_Enhanced"])

# Print silhouette scores
print(f"Silhouette Score for Original K-Means (Euclidean): {silhouette_original:.4f}")
print(f"Silhouette Score for Enhanced K-Means (LOF): {silhouette_enhanced:.4f}")

### --- USER CLUSTERING SUMMARY ---
# Identify outlier userIds
outlier_user_ids = user_genre_ratings[outliers_lof == -1].index.tolist()

# Print summary
print("\nUser Clustering Summary:")
print("  Total users:", user_genre_ratings.shape[0])
print("  Inliers after LOF filtering:", user_genre_ratings_filtered.shape[0])
print("  Outliers removed:", user_genre_ratings.shape[0] - user_genre_ratings_filtered.shape[0])
print("  Outlier userIds (LOF detected):", outlier_user_ids)

