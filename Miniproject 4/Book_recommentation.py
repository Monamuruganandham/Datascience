import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import Counter

# Load dataset
df = pd.read_csv(r"C:\Users\Mohan\PycharmProjects\pythonProject\Miniproject4\Audible_Catalog_Cleaned_with_Clusters.csv")

# Sidebar for user preferences

st.title("📚 Book Recommendation System")
st.sidebar.title("Customize Your Book Journey")
selected_genre = st.sidebar.selectbox("Select your favorite genre:", df["Ranks and Genre"].unique())
#selected_book = st.sidebar.text_input("Enter a book you like:")
selected_book = st.selectbox("Enter a book you like:",df["Book Name"])


def recommend_sci_fi_books(df, n=5):
    sci_fi_books = df[df["Ranks and Genre"].str.contains("Science Fiction", case=False, na=False)]
    if sci_fi_books.empty:
        return "No Science Fiction books found in the dataset."
    top_sci_fi_books = sci_fi_books.sort_values(by=["Rating", "Number_of_Reviews"], ascending=[False, False]).head(n)
    return top_sci_fi_books[["Book Name", "Author", "Rating", "Number_of_Reviews"]]

# Create tabs
tab1, tab2 ,tab3,tab4= st.tabs(["📖 Recommended Books", "📊 General Insights","\U0001F52E Thriller Insights","\U0001F680 Science Fiction Picks"])

# TF-IDF Vectorization for Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
df['Description'] = df['Description'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['Description'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to find the best book match
def find_best_match(book_name, df):
    book_list = df["Book Name"].tolist()
    best_match, score = process.extractOne(book_name, book_list)
    if score > 60:
        return best_match
    return None

# Function to recommend books using a hybrid approach
def hybrid_recommend(book_name, df, similarity_matrix, n=5):
    matched_book = find_best_match(book_name, df)
    if matched_book is None:
        return "Book not found. Please try another title."

    book_idx = df[df["Book Name"] == matched_book].index[0]

    # Content-Based Recommendations
    similar_books = list(enumerate(similarity_matrix[book_idx]))
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:n + 1]
    content_recommendations = [df.iloc[i[0]][['Book Name', 'Author']] for i in similar_books]

    # Clustering-Based Recommendations
    book_cluster = df.loc[book_idx, 'Cluster']
    cluster_recommendations = df[df['Cluster'] == book_cluster].sample(n)

    return pd.concat([pd.DataFrame(content_recommendations), cluster_recommendations[['Book Name', 'Author']]])

# Recommended Books Tab
with tab1:
    st.subheader("📖 Recommended Books")
    if selected_book:
        recommendations = hybrid_recommend(selected_book, df, similarity_matrix)
    else:
        recommendations = df[df["Ranks and Genre"] == selected_genre][["Book Name", "Author", "Rating"]].sample(5)
    st.write(recommendations)

# EDA Tab
with tab2:
    st.subheader("📊 Data Insights")

    # Top 10 Most Common Genres
    st.subheader("Top 10 Most Common Genres")
    fig, ax = plt.subplots(figsize=(10, 5))
    df["Ranks and Genre"].value_counts().nlargest(10).plot(kind="bar", color="green", ax=ax)
    ax.set_title("Top 10 Most Common Genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Highest Rated Authors
    st.subheader("Top Rated Authors")
    top_authors = df.groupby("Author")["Rating"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_authors)

    # Rating Distribution
    st.subheader("⭐ Rating Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Rating"], bins=10, kde=True, color='blue')
    st.pyplot(plt)

    # Heatmap for Ratings & Reviews Correlation
    st.subheader("📈 Ratings vs Reviews Correlation")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[["Rating", "Number_of_Reviews"]].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Ratings vs. Number of Reviews
    st.subheader("Ratings vs. Number of Reviews")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x="Number_of_Reviews", y="Rating", alpha=0.5)
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Book Rating")
    ax.set_title("How Ratings Vary with Review Counts")
    st.pyplot(fig)

    # Book Clustering Based on Descriptions
    st.subheader("Book Clustering Based on Descriptions")
    df_filtered = df.dropna(subset=["Description", "Book Name"]).head(30)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(df_filtered["Description"])
    Z = linkage(X.toarray(), method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, labels=df_filtered["Book Name"].values, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Book Clusters Based on Descriptions")
    st.pyplot(fig)

    # Genre Similarity Heatmap
    st.subheader("Genre Similarity Heatmap")
    top_genres = df["Ranks and Genre"].str.split(",").explode().value_counts().index[:10]
    genre_matrix = pd.DataFrame(0, index=top_genres, columns=top_genres)
    for genres in df["Ranks and Genre"].dropna():
        genre_list = genres.split(",")
        for g1 in genre_list:
            for g2 in genre_list:
                if g1 in top_genres and g2 in top_genres:
                    genre_matrix.loc[g1, g2] += 1
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(genre_matrix, annot=True, cmap="coolwarm", fmt="d", ax=ax)
    ax.set_title("Genre Similarity Heatmap")
    st.pyplot(fig)

    # Effect of Author Popularity on Ratings
    st.subheader("Effect of Author Popularity on Ratings")
    popular_authors = df.groupby("Author")["Number_of_Reviews"].sum().nlargest(20).index
    df["Author Popularity"] = df["Author"].apply(lambda x: "Popular" if x in popular_authors else "Less Popular")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Author Popularity", y="Rating", ax=ax)
    ax.set_title("Ratings by Author Popularity")
    st.pyplot(fig)

    # Feature Importance in Rating Prediction
    st.subheader("Feature Importance in Book Recommendation")
    features = ["Listening Time", "Number_of_Reviews", "Price"]
    df[features] = df[features].replace("Unknown", pd.NA)
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    X = df[features].fillna(df[features].median())
    y = df["Rating"].fillna(df["Rating"].mean())
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=features)
    fig, ax = plt.subplots(figsize=(8, 4))
    feature_importance.sort_values().plot(kind="barh", color="blue", ax=ax)
    ax.set_title("Feature Importance in Rating Prediction")
    st.pyplot(fig)

    # Hidden Gems: Highly Rated but Low Popularity Books
    st.subheader("Hidden Gems: Highly Rated, Low Popularity Books")
    rating_threshold = 4.5
    popularity_threshold = 50
    hidden_gems_books = df[(df["Rating"] >= rating_threshold) & (df["Number_of_Reviews"] <= popularity_threshold)]
    if not hidden_gems_books.empty:
        st.dataframe(hidden_gems_books[["Book Name", "Rating", "Number_of_Reviews", "Ranks and Genre"]])
    else:
        st.write("No hidden gems found based on the specified criteria.")

with tab3:
    st.subheader("\U0001F52E Insights for Thriller Lovers")
    thriller_books = df[df["Ranks and Genre"].str.contains("Thriller", case=False, na=False)]
    st.write("Number of Thriller Books:", thriller_books.shape[0])
    # Ratings Distribution for Thriller Books
    st.subheader("Ratings Distribution for Thriller Books")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(thriller_books["Rating"], bins=10, kde=True, color='red', ax=ax)
    st.pyplot(fig)

    # Popular Thriller Authors
    st.subheader("Top 10 Thriller Authors")
    top_thriller_authors = thriller_books["Author"].value_counts().head(10)
    st.bar_chart(top_thriller_authors)

    # Highly Rated Thriller Books with Low Reviews (Hidden Gems)
    st.subheader("Hidden Gems: Highly Rated but Less Popular Thrillers")
    hidden_thrillers = thriller_books[(thriller_books["Rating"] >= 4.5) & (thriller_books["Number_of_Reviews"] < 50)]
    if not hidden_thrillers.empty:
        st.dataframe(hidden_thrillers[["Book Name", "Author", "Rating", "Number_of_Reviews"]])
    else:
        st.write("No hidden gems found based on the specified criteria.")

with tab4:
    st.subheader("\U0001F680 Top 5 Science Fiction Books")
    top_sci_fi_books = recommend_sci_fi_books(df)
    st.write(top_sci_fi_books)










