import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Numerical features for clustering
numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

# Preprocess
df = df.dropna(subset=numerical_features)
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Standardize features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Recommendation function
def recommend_songs(song_name, df, num_recommendations=5):
    if song_name not in df['track_name'].values:
        return pd.DataFrame({"Message": [f"'{song_name}' not found."]})

    # Get the cluster for the input song
    song_cluster = df[df['track_name'] == song_name]['Cluster'].values[0]

    # Filter songs from the same cluster
    same_cluster_songs = df[df['Cluster'] == song_cluster].reset_index(drop=True)

    # Get index of the input song within this filtered list
    song_index = same_cluster_songs[same_cluster_songs['track_name'] == song_name].index[0]

    # Calculate cosine similarity
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    # Get top N recommendations (excluding the song itself)
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][['track_name', 'artist_name', 'genre']]

    return recommendations


# Streamlit UI
st.title("🎧 Spotify Song Recommender")

song_input = st.text_input("Enter a song name:")


# 🎼 Genre Filter
genres = df['genre'].unique()
selected_genre = st.selectbox("🎼 Filter by Genre", sorted(genres))

# Filter dataset by selected genre
filtered_df = df[df['genre'] == selected_genre]

# 🎵 Song Selection based on filtered genre
song_input = st.selectbox("🎵 Choose a song:", sorted(filtered_df['track_name'].unique()))

# 🔢 Number of recommendations
num_recs = st.slider("📊 Number of recommendations", 1, 10, 5)

# 🔘 Recommend button
if st.button("Recommend"):
    recs = recommend_songs(song_input, filtered_df, num_recommendations=num_recs)
    st.subheader(f"🎯 Songs similar to **{song_input}**:")
    st.dataframe(recs)

# 📈 Cluster Distribution Chart (optional)
st.subheader("📊 Number of Songs per Cluster")
cluster_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)
