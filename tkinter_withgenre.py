import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import calinski_harabasz_score, pairwise_distances

# --- Load datasets ---
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# --- Extract genres ---
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# --- Merge ratings with movies ---
df = ratings.merge(movies, on='movieId')

# --- Compute average ratings per genre per user ---
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# --- Standardize data ---
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

# --- Remove outliers using LOF ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, metric="canberra")
outliers_lof = lof.fit_predict(user_genre_ratings_scaled)
user_genre_ratings_filtered = user_genre_ratings[outliers_lof == 1]
user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

# --- Custom KMeans using Canberra distance ---
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

# --- Determine best k using Calinski-Harabasz Index ---
best_k = None
best_score = -np.inf
for k in range(2, 6):
    labels, _ = custom_kmeans_canberra(user_genre_ratings_filtered_scaled, n_clusters=k)
    score = calinski_harabasz_score(user_genre_ratings_filtered_scaled, labels)
    if score > best_score:
        best_k = k
        best_score = score

# --- Final KMeans clustering ---
clusters_enhanced, _ = custom_kmeans_canberra(user_genre_ratings_filtered_scaled, n_clusters=best_k)
user_genre_ratings_filtered["Cluster_Enhanced"] = clusters_enhanced

# --- Recommendation Function ---
def recommend_movies_for_user(user_id):
    try:
        user_id = int(user_id)
        if user_id not in user_genre_ratings_filtered.index:
            return None, [], []

        user_cluster = user_genre_ratings_filtered.loc[user_id, 'Cluster_Enhanced']
        similar_users = user_genre_ratings_filtered[user_genre_ratings_filtered['Cluster_Enhanced'] == user_cluster].index

        user_movies = df[df['userId'] == user_id]
        highly_rated_movies = user_movies[user_movies['rating'] >= 4.0].sort_values(by="rating", ascending=False)
        highly_rated_titles = highly_rated_movies['title'].tolist()[:10]

        # Movies rated by users in the same cluster but not yet rated by the target user
        cluster_movies = df[(df['userId'].isin(similar_users)) & (~df['movieId'].isin(user_movies['movieId']))]
        recommended_movie_ids = cluster_movies.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10).index
        recommended_titles = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

        return user_cluster, highly_rated_titles, recommended_titles
    except Exception:
        return None, [], []

# --- Tkinter App ---
def launch_app():
    app = tk.Tk()
    app.title("🎬 Movie Recommendations (Clustering-Based)")
    app.geometry("700x600")
    app.configure(bg="white")

    # --- Top Frame for Dropdown ---
    top_frame = tk.Frame(app, bg="white")
    top_frame.pack(pady=20)

    tk.Label(top_frame, text="Select User ID:", font=("Helvetica", 12, "bold"), bg="white").grid(row=0, column=0, padx=10)
    
    user_ids = list(user_genre_ratings_filtered.index)
    selected_user = tk.StringVar()
    user_menu = ttk.Combobox(top_frame, textvariable=selected_user, values=user_ids, state="readonly", width=10)
    user_menu.grid(row=0, column=1)

    # --- Button Frame ---
    button_frame = tk.Frame(app, bg="white")
    button_frame.pack(pady=10)

    def show_recommendations():
        user_id = selected_user.get()
        if not user_id:
            messagebox.showwarning("Missing selection", "Please select a user ID.")
            return

        cluster, high_rated, recommended = recommend_movies_for_user(user_id)
        result_text.config(state="normal")
        result_text.delete(1.0, tk.END)

        if cluster is None:
            result_text.insert(tk.END, f"User {user_id} not found in filtered dataset.\n")
        else:
            result_text.insert(tk.END, f"📍 User ID: {user_id}\n")
            result_text.insert(tk.END, f"🧠 Cluster: {cluster}\n\n")

            result_text.insert(tk.END, "📌 Top Rated by User:\n")
            for title in high_rated:
                result_text.insert(tk.END, f" - {title}\n")

            result_text.insert(tk.END, "\n🎥 Recommended Movies:\n")
            for title in recommended:
                result_text.insert(tk.END, f" - {title}\n")

        result_text.config(state="disabled")

    tk.Button(button_frame, text="🎯 Get Recommendations", command=show_recommendations,
              font=("Helvetica", 11), bg="#4CAF50", fg="white", padx=15, pady=5).pack()

    # --- Results Text Box ---
    result_frame = tk.Frame(app, bg="white")
    result_frame.pack(pady=10, fill="both", expand=True)

    result_text = tk.Text(result_frame, height=25, wrap="word", font=("Courier", 10), bg="#f9f9f9", relief="groove", bd=2)
    result_text.pack(padx=10, pady=10, fill="both", expand=True)
    result_text.config(state="disabled")

    app.mainloop()


# --- Launch the app ---
launch_app()
