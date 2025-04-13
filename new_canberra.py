import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import format
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Define genres and assign binary flags
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with movies
df = ratings.merge(movies, on='movieId')

# Compute user average ratings per genre
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Original"] = clusters_original

### --- CUSTOM K-MEANS USING CANBERRA DISTANCE ---
def custom_kmeans_canberra(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric='canberra')
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers

# Apply Enhanced K-Means
chosen_k = 5
clusters_enhanced, _ = custom_kmeans_canberra(user_genre_ratings_scaled, n_clusters=chosen_k)
user_genre_ratings["Cluster_Enhanced"] = clusters_enhanced

### --- SILHOUETTE SCORE CALCULATION ---
# Calculate silhouette scores
silhouette_original = format(silhouette_score(user_genre_ratings_scaled, clusters_original, metric='euclidean')) 
silhouette_enhanced = format(silhouette_score(user_genre_ratings_scaled, clusters_enhanced, metric='canberra'), reference_value=silhouette_original)

print(f"Silhouette Score (Original K-Means - Euclidean): {silhouette_original}")
print(f"Silhouette Score (Enhanced K-Means - Canberra): {silhouette_enhanced}")

### --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(user_genre_ratings_scaled)
user_genre_ratings["PCA1"] = pca_result[:, 0]
user_genre_ratings["PCA2"] = pca_result[:, 1]

### --- PLOT PCA SCATTERPLOTS WITH GENRE AXIS CONTRIBUTIONS ---
def plot_pca_clusters(data, pca_model, feature_names):
    # Get component loadings
    components = pd.DataFrame(pca_model.components_, columns=feature_names, index=["PC1", "PC2"])

    # Get top 5 contributing genres per PC
    top_pc1 = components.loc["PC1"].abs().sort_values(ascending=False).head(5).index.tolist()
    top_pc2 = components.loc["PC2"].abs().sort_values(ascending=False).head(5).index.tolist()

    # Create axis labels
    xlabel = f"{', '.join(top_pc1)}"
    ylabel = f"{', '.join(top_pc2)}"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cluster in sorted(data["Cluster_Original"].unique()):
        subset = data[data["Cluster_Original"] == cluster]
        axes[0].scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    axes[0].set_title("Original K-Means (Euclidean)")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    for cluster in sorted(data["Cluster_Enhanced"].unique()):
        subset = data[data["Cluster_Enhanced"] == cluster]
        axes[1].scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    axes[1].set_title("Enhanced K-Means (Canberra)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Print full genre contributions
    print("\nFull Genre Contributions to PCA Components:")
    print(components.T.sort_values(by="PC1", ascending=False))

# Plot with genre contributions in axis labels
plot_pca_clusters(user_genre_ratings, pca, genres)
