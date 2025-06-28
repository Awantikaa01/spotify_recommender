import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

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
def recommend_songs(song_name, data, num_recommendations=5):
    if song_name not in data['track_name'].values:
        return pd.DataFrame({"Message": [f"'{song_name}' not found."]})

    song_cluster = data[data['track_name'] == song_name]['Cluster'].values[0]
    same_cluster_songs = data[data['Cluster'] == song_cluster].reset_index(drop=True)

    song_index = same_cluster_songs[same_cluster_songs['track_name'] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][['track_name', 'artist_name', 'genre']]

    return recommendations

# Streamlit UI
st.set_page_config(page_title="Spotify Recommender", layout="centered")
st.title("🎧 Spotify Song Recommender")
st.markdown("Select a genre and song to get personalized music suggestions.")

# 🎼 Genre Filter
genres = df['genre'].dropna().unique()
selected_genre = st.selectbox("🎼 Filter by Genre", sorted(genres))
filtered_df = df[df['genre'] == selected_genre]

# 🎵 Song Selection
song_input = st.selectbox("🎵 Choose a song:", sorted(filtered_df['track_name'].unique()))

# 🔢 Number of recommendations
num_recs = st.slider("📊 Number of recommendations", 1, 10, 5)

# 🔘 Recommend button
if st.button("🔍 Recommend"):
    recs = recommend_songs(song_input, filtered_df, num_recommendations=num_recs)
    st.subheader(f"🎯 Songs similar to **{song_input}**:")
    st.dataframe(recs)

    # 🔊 Play audio preview if files available
    audio_path = f"songs/{song_input}.mp3"
    try:
        st.audio(audio_path)
    except:
        st.warning("🔇 Audio preview not available for this track.")

# 📈 Cluster Distribution Chart
st.subheader("📊 Cluster Distribution")
cluster_counts = df['Cluster'].value_counts().sort_index()
fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
             labels={'x': 'Cluster', 'y': 'Number of Songs'},
             title="Distribution of Songs Across Clusters",
             color=cluster_counts.values)
st.plotly_chart(fig)

# 📥 Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and scikit-learn. | Developed by Awantika")
