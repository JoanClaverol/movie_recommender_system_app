import pandas as pd
import numpy as np
import streamlit as st
# import os
# os.chdir('8-Recommender-Systems/app/')
import module

# -- Load data
path = "./data/ml-latest-small/"

links = pd.read_csv(path + "links" + ".csv")
movies = pd.read_csv(path + "movies" + ".csv")
ratings = pd.read_csv(path + "ratings" + ".csv")
rdy_rat = ratings.drop(columns="timestamp")
links = pd.read_csv(path + "tags" + ".csv")

# personal rating 
pers_path = "./data/collected-ratings/"
joan = pd.read_csv(pers_path + "joan_ratings.csv").assign(userId = 999)
nuria = pd.read_csv(pers_path + "nuria_ratings.csv").assign(userId = 998)
rdy_rat = pd.concat([rdy_rat, joan, nuria])

# -- Config the page
app_title = "WBSFLIX"
st.set_page_config(
    page_title=app_title,
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Title the app
st.title("WBSFLIX")

# -- Define inputs on sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("# Parameters")

# -- INPUTS
num_movies = st.sidebar.text_input("Enter the number of movies:")
movie_name = st.sidebar.text_input("Enter the name of a movie:").lower()
user_id = st.sidebar.text_input("Enter userId:")

# -- RECOMMENDER
st.markdown("## Recommendation(s) for you:")
# -- Popular movies
if not num_movies and not movie_name and not user_id:
    pop_movies = module.popularity_recommender(rdy_rat, n_pop_movies=5)[
        "item"
    ].tolist()
    _ = [st.text(t) for t in module.item_to_movie_title(pop_movies, movies)]
# -- Popular movies (n defined)
elif num_movies and not movie_name and not user_id:
    pop_movies = module.popularity_recommender(rdy_rat, n_pop_movies=int(num_movies))[
        "item"
    ].tolist()
    _ = [st.text(t) for t in module.item_to_movie_title(pop_movies, movies)]
# -- Similar movies based on MOVIE
elif num_movies and movie_name and not user_id:
    fitlered_movies = (
        movies.assign(check=lambda x: x["title"].str.lower().str.contains(movie_name))
        .query("check")
        .sort_values("title")
    )
    movies_titles = fitlered_movies["title"].tolist()
    if len(movies_titles) > 1:
        st.markdown("#### First, specify the movie name (exact name):")
        for movie in movies_titles:
            st.text(movie)
    elif len(movies_titles) == 1:
        movies_title = movies_titles[0]
        movieId = fitlered_movies.query("title == @movies_title")["movieId"].values[0]
        recommendations = module.item_based_recommender(
            rdy_rat, movieId, 
            n=int(num_movies)
        )
        _ = [st.text(t) for t in module.item_to_movie_title(recommendations, movies)]
    else: 
        st.text('Movie does not exist in the database')
# -- Similar movies based on USER
elif num_movies and not movie_name and user_id:
    recommendations = module.get_user_based_recommendations(
        rdy_rat, int(user_id), top_n=int(num_movies), min_n_ratings=50, neighbours=7
    )
    if recommendations is False:
        st.text("User not found!")
    for t in module.item_to_movie_title(recommendations, movies):
        st.text(t)
# -- Similar movies based on MOVIE and USER
elif num_movies and movie_name and user_id: 
    # find out movies our user id has not seen
    user_id = int(user_id)
    movies_seen = rdy_rat.query("userId == @user_id")['movieId'].tolist()
    movies_not_seen = movies.query("movieId != @movies_seen")['movieId'].tolist()
    # filter them out
    fitlered_movies = (
    movies
        .assign(check=lambda x: x["title"].str.lower().str.contains(movie_name))
        .query("check")
        .sort_values("title")
    )
    movies_titles = fitlered_movies["title"].tolist()
    if len(movies_titles) > 1:
        st.markdown("#### First, specify the movie name (exact name):")
        for movie in movies_titles:
            st.text(movie)
    elif len(movies_titles) == 1:
        movies_title = movies_titles[0]
        movieId = fitlered_movies.query("title == @movies_title")["movieId"].values[0]
        recommendations = module.item_based_recommender(
            rdy_rat.query("movieId == @movies_not_seen"), 
            movieId, 
            n=int(num_movies)
        )
        _ = [st.text(t) for t in module.item_to_movie_title(recommendations, movies)]
    else: 
        st.text('Movie does not exist in the database')