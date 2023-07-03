import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def item_to_movie_title(item_ids: list, dictionary):
    """
    Given a list of item ids, return a list of the corresponding 
    movie titles.
    """
    return dictionary.query("movieId == @item_ids")["title"].tolist()


def dense_df_preparation(data: pd.DataFrame):
    """
    Given a dataframe, check if the df only contains three columns. If so, 
    return a df with renamed columns in the following way: 
        ['user','item','rating']

    In case it contains more than three columns, return false
    Example: movies ratings only should include columns userId,	movieId and rating, sorted in that way. 
    """
    # quick check number of columns
    if len(data.columns) != 3:
        print(
            """
            Be sure to have added a dataframe with only the columns: 
                users, items and ratings
            They have to be sorted in the same way!
        """
        )
        return False

    # define column names
    data.columns = ["user", "item", "rating"]

    return data


def movie_n_ratings_filter(data: pd.DataFrame, min_n_ratings=100):
    """
    Given ratings dataframe, return the a list of movies Ids that has been
    rated a specifici amount of times. 
    """
    filtered_movies_id = (
        data.groupby("item")["item"]
        .count()
        .reset_index(name="count")
        .query("count > @min_n_ratings")
        ["item"].tolist()
    )
    return filtered_movies_id


def popularity_recommender(data: pd.DataFrame, min_n_ratings=100, n_pop_movies=5):
    """
    Given a dense dataframe and a minimum number of ratings, return
    a dataframe with the highest rated films on average. 

    FURTHER IMPROVEMENTS: 
    The popularity-based recommender should include, at least, all of 
    these genres:
        Comedy
        Drama
        Thriller
        Sci-Fi
        Children
    The similarity-based recommender should include, at least, 
    two movies that share a genre with the inputted one.
    """
    # prepare our dense matrix
    dense_df = dense_df_preparation(data)
    # in case there is a problem, stop the recommender
    if dense_df is False:
        return
    # select movies with a minimum number of ratings
    filtered_movies = movie_n_ratings_filter(dense_df, min_n_ratings)
    # find most popular movies
    recommended_movies = (
        dense_df
        # filter non popular films
        .query("item == @filtered_movies")
        # find the average rating and total number of ratings
        # by each movie
        .groupby("item")
        .agg(avg_rating=("rating", "mean"))
        # sort ratings from higher to lower
        .sort_values("avg_rating", ascending=False)
        # number of most popular movies to return
        .head(n_pop_movies)
        .reset_index()
        # remove innecessary decimals on the average rating
        .assign(avg_rating=lambda x: round(x["avg_rating"], 2))
    )

    return recommended_movies


def popularity_chat_bot(data, dictionary):
    print("Hi! I'm your personal recommender. Let me recommend you a film!.")
    # execute recommender
    recommendations = popularity_recommender(data.drop(columns="timestamp"))
    # get movie title
    popular_movie = item_to_movie_title(recommendations["item"].tolist(), dictionary)[0]
    print(f"You will probably like {popular_movie}")


def movie_id_finder(dictionary):
    """
    Given a title, find the closest movie title and return its id
    """
    # insert a movie title
    title = input().lower()
    # look if there is any movie containing this title, if so, filter them
    fitlered_movies = (
        dictionary.assign(check=lambda x: x["title"].str.lower().str.contains(title))
        .query("check")
        .sort_values("title")
    )
    # if there are multiple movies containing the same title, ask to which
    # one is the user refering. Once it is specified, filter all the others out
    if fitlered_movies.shape[0] > 1:
        print("Which one of the following movies do you mean? ")
        n_movie = 1
        movies_titles = fitlered_movies["title"].tolist()
        for movie in movies_titles:
            print("\t" + movie + " [type " + str(n_movie) + "]")
            n_movie += 1
        # ask about the movie
        n_movie = input()
        selected_title = movies_titles[int(n_movie) - 1]
        fitlered_movies = fitlered_movies.query("title == @selected_title")
    # if there is no movies with that title, return False
    elif fitlered_movies.shape[0] == 0:
        print("No movies has been found")
        return False
    # inform about the movie selected
    print(f"The selected movie is {fitlered_movies['title'].values[0]}")
    return fitlered_movies["movieId"].values[0]


def sparse_df_preparation(data: pd.DataFrame):
    """
    Given a data return an sparse matrix with index user, columns items
    and values rating
    """
    # prepare our dense matrix
    dense_df = dense_df_preparation(data)
    # in case there is a problem, stop the recommender
    if dense_df is False:
        return False
    # create the sparsed dataframe
    sparse_df = dense_df.pivot("user", "item", "rating")

    return sparse_df


import warnings


def item_based_recommender(
    data: pd.DataFrame, item, n=5, min_n_ratings=80, n_pop_movies=4000
):
    """
    Given a data and an item id, return the n most correlated items 

    FURTHER IMPROVEMENT: 
    Need to select only items that has a descent number of reviews to avoid 
    higher correlations with movies that have few reviews. 
    """
    # get popular films
    dense_df = dense_df_preparation(data)
    pop_items = popularity_recommender(dense_df, min_n_ratings, n_pop_movies)

    print(f"Total number of films compared with {len(pop_items['item'])}")
    # get the sparsed matrix
    sparse_df = sparse_df_preparation(data)
    if sparse_df is False:
        return
    # do not show warnings
    warnings.filterwarnings("ignore")
    # use the pairwise function to find the most correlated movies to
    # the movie we specified
    correlated_items = sparse_df[pop_items["item"]].corrwith(sparse_df[item])
    # check if there is no correlation to any other film, if so,
    # return a message
    if np.isnan(correlated_items.values).all():
        print("No correlation with any other film!")
        return
    # if there is correlation with other movies:
    else:
        top_corr_item = (
            correlated_items
            # sort them from the highest correlated to the lowers
            .sort_values(ascending=False).reset_index()
            # filter the same movie
            .query("item != @item")
            # select the top n correlated movies
            .head(n)
        )
        # return the column item and transform it into a list
        return top_corr_item["item"].tolist()


def item_based_chat_bot(data, dictionary):
    print("Hi! I'm your personal recommender. Tell me a film you've liked.")
    # help user to find the movie name
    movie_id = movie_id_finder()
    # find out the most correlated films to the given movie id
    recommendations = item_based_recommender(
        data.drop(columns="timestamp"), item=movie_id
    )
    if not recommendations:
        return
    # get top movie title
    recommended_movie = item_to_movie_title(recommendations, dictionary)[0]
    print(f"You will probably like {recommended_movie}")


from sklearn.model_selection import train_test_split


def train_test_creation(data, random_state=42, train_size=0.8):
    """
    Given a ratings data, transform it into a sparse matrix and return
    a train, test, train_pos, test_pos (in that order)
    """
    # create the sparse matrix
    sparse_df = sparse_df_preparation(data)
    # locate all positions with a rating
    ratings_pos = pd.DataFrame(
        # find all the positions different than missing values
        # 1. transform the sparse matrix into a numpy array
        # 2. the ~ operator help us to define "not". If we combine it with
        #   np.isnan we are saying "is not nan"
        # 3. np.argwhere help us to find the positions with the argument
        #   we are looking for in an array. In that case is all the non
        #   nan positions in our array
        np.argwhere(~np.isnan(np.array(sparse_df)))
    )  # np.argwhere(a) is almost the same as np.transpose(np.nonzero(a)),
    # but produces a result of the correct shape for a 0D array.
    # Source: numpy documentation

    train_pos, test_pos = train_test_split(
        ratings_pos, random_state=random_state, train_size=train_size
    )

    # create an empty dataframe full of 0, with the same shape as the sparse_df data
    train = np.zeros(sparse_df.shape)
    # fill the set with the sparse_df ratings based on the train positions
    for pos in train_pos.values:
        index = pos[0]
        col = pos[1]
        train[index, col] = sparse_df.iloc[index, col]
    train = pd.DataFrame(train, columns=sparse_df.columns, index=sparse_df.index)

    # now it is time for the test set. We will follow the same process
    test = np.zeros(sparse_df.shape)
    for pos in test_pos.values:
        index = pos[0]
        col = pos[1]
        test[index, col] = sparse_df.iloc[index, col]
    test = pd.DataFrame(test, columns=sparse_df.columns, index=sparse_df.index)

    return train, test, train_pos, test_pos


def user_based_recommender(index_name, column_name, sim_df, sparse_df):
    results = (
        pd.DataFrame(
            {
                "ratings": sparse_df.loc[:, column_name],
                "similitudes": sim_df.loc[index_name, :].tolist(),
            }
        )
        .query("ratings != 0")
        .assign(weighted_ratings=lambda x: x.ratings * x.similitudes)
        .agg({"weighted_ratings": "sum", "similitudes": "sum"})
    )
    pred_rating = results["weighted_ratings"] / results["similitudes"]
    return pred_rating

def nearest_neighbours_user_based_recommender(
    index_name, column_name, sim_df, sparse_df, neighbours=10
    ):
    results = (
        pd.DataFrame(
            {
                "ratings": sparse_df.loc[:, column_name],
                "similitudes": sim_df.loc[index_name, :].tolist(),
            }
        )
        .query("ratings != 0")
        .head(neighbours)
        .assign(weighted_ratings=lambda x: x.ratings * x.similitudes)
        .agg({"weighted_ratings": "sum", "similitudes": "sum"})
    )
    pred_rating = results["weighted_ratings"] / results["similitudes"]
    return pred_rating


def round_off_rating(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""
    return round(number * 2) / 2


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def predictions_analysis(y_true, preds):
    print(
        f"""
        MSE: {mean_squared_error(y_true, preds)}
        RMSE: {mean_squared_error(y_true, preds)**0.5}
        MAE: {mean_absolute_error(y_true, preds)}
        MAPE: {mean_absolute_percentage_error(y_true, preds)}
        """
    )

    p_df = (
        pd.DataFrame({"preds": preds, "true": y_true})
        .groupby(["preds", "true"])["preds"]
        .count()
        .groupby(level=0)
        .apply(lambda x: 100 * x / float(x.sum()))
        .reset_index(name="count_perc")
    )  # .reset_index(name='count')
    print("Head() plot data:")
    print(p_df.head(10))
    plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=p_df,
        x="preds",
        y="true",
        size="count_perc",
        hue="count_perc",
        sizes=(20, 2000),
    )
    sns.lineplot(x=[0, 5], y=[0, 5], color="lightgrey")
    plt.title(
        "Percentage of observations by each predicted\n rating compared to true rating"
    )
    plt.xlabel("Predicted ratings")
    plt.ylabel("True ratings")
    plt.legend([], [], frameon=False)
    sns.despine()
    plt.show()


from sklearn.metrics.pairwise import cosine_similarity


def get_user_based_recommendations(
    data, user, top_n=5, min_n_ratings=6, n_pop_movies=4000, neighbours=10
):
    """
    Given the rating data and a user id, return the top n films

    FURTHER IMPROVEMENT: 
    Increase the performance by excluding all the films with a low
    number of ratings. 
    An option would be to combine the popular and user-based recommender.
    """
    # get popular films
    dense_df = dense_df_preparation(data)
    pop_items = popularity_recommender(dense_df, min_n_ratings, n_pop_movies)
    print(f"Total number of films compared with {len(pop_items['item'])}")
    # get the sparse matrix
    sparse_df = sparse_df_preparation(data).replace(np.nan, 0)
    if user not in sparse_df.index:
        print("User not found!")
        return False
    sparse_df = sparse_df[pop_items["item"]]
    # calculate the similitude matrx with cosine strategy
    similitudes_df = pd.DataFrame(
        cosine_similarity(sparse_df), columns=sparse_df.index, index=sparse_df.index
    )
    # find out the non rated movies by our user
    non_rated_movies = (
        sparse_df.loc[user, :].reset_index(name="user_rating").query("user_rating == 0")
    )
    # find the predictied for each non rated film
    # display a nice progress bar to know how the process is going
    pred_rat = []
    i = 0
    n = len(non_rated_movies["item"])
    for item in non_rated_movies["item"].tolist():
        pred_rat.append(
            nearest_neighbours_user_based_recommender(
                user, item, similitudes_df, sparse_df, neighbours
                )
            )
        # progress bar
        sys.stdout.write("\r")
        j = (i + 1) / n
        sys.stdout.write("[%-20s] %d%%" % ("=" * int(20 * j), 100 * j))
        sys.stdout.flush()
        i += 1
    # add the predicted ratings and sort them
    top_pred_ratings = (
        non_rated_movies.assign(pred=pred_rat)
        .sort_values("pred", ascending=False)
        .head(top_n)
    )
    print(top_pred_ratings)
    # return a list of the movies id with a higer predicted rating
    return top_pred_ratings["item"].tolist()


def user_based_chat_bot(data):
    print("Hi! I'm your personal recommender. Can you remember me your user id?")
    # help user to find the movie name
    user_id = int(input())
    # find out the most correlated films to the given movie id
    print("Based on your previous ratings, let me recommend you one film!")
    recommendations = get_user_based_recommendations(
        data.drop(columns="timestamp"), user=user_id, top_n=5
    )
    if not recommendations:
        return
    # get top movie title
    recommended_movie = item_to_movie_title(recommendations)  # [0]
    print(f"\nYou will probably like {recommended_movie}")

def rating_evaluator(): 
    """
    Given a rating, check if it's empty so return a NaN, or round it
    to +- 0.5 if it's decimal. 
    """
    value = input()
    if value == "":
        return np.nan
    else:
        value = round_off_rating(float(value))
        return value


def movie_rater(movies, ratings, path, min_n_ratings=50, n_movies_to_rate=10):
    """
    Given the movies and ratings help a user to rate random films with a minimum
    number of ratings.
    To store the ratings, a path should be provided. 
    """
    print("Please, introduce your user id: ")
    user = int(input())

    movies_to_rate = (
        pd.DataFrame({"item": movie_n_ratings_filter(ratings, min_n_ratings)})
        .sample(n_movies_to_rate)
        .assign(
            title=lambda x: item_to_movie_title(
                x["item"].tolist(), dictionary=movies
            ),
            user=user,
        )
    )

    movies = movies_to_rate["title"].tolist()
    user_ratings = []
    print("Rate the following films from 0.5 to 5:")
    for movie in movies:
        sys.stdout.write("\r")
        sys.stdout.write(movie)
        print("")
        user_ratings.append(rating_evaluator())
        # val = input()
        # if val == "":
        #     user_ratings.append(np.nan)
        # else:
        #     val = round_off_rating(float(val))
        #     user_ratings.append(val)

    movies_to_rate = movies_to_rate.assign(ratings=user_ratings)
    print(movies_to_rate)

    print("Do you you want to save this ratings? (y/n)")
    save = input()
    if save == "y":
        print("Please, provide a file name:")
        file_name = input()
        path = path + file_name + ".csv"
        movies_to_rate.to_csv(path, index=False)
        print(f"Data saved as {path}.")
    else:
        print("Data not saved.")



def new_movie_rating(movies): 
    """
    Given a text and a dictionary of all the relation between movies and
    movieId, return a movieId and a rating for the selected movie
    """
    print('Please, introduce the movie title: ')
    movie_id = movie_id_finder(movies)

    print('Introduce the rating: ')
    rating = rating_evaluator()

    return movie_id, rating

def new_movies_ratings_adder(movies): 
    """
    Given all the movies, add as many movies and its rating as the user wants to, 
    and return a dataframe with movieId and rating of all the rated films. 
    """
    # lists to store the output
    new_movies_id, new_movie_ratings = [], []
    # add the movie you want to rate
    movie_id, rating = new_movie_rating(movies)
    new_movies_id.append(movie_id)
    new_movie_ratings.append(rating)
    # check if there are more movies that want to be rated
    print("Do you want to add a rating to another film?(y/n)")
    answer = input()
    while answer == "y": 
        movie_id, rating = new_movie_rating(movies)
        new_movies_id.append(movie_id)
        new_movie_ratings.append(rating)
        print("Do you want to add a rating to another film?(y/n)")
        answer = input()

    return pd.DataFrame({
        "movieId":new_movies_id, 
        "rating":new_movie_ratings
    })