import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
print(movies_df.head())

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head())

movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

moviesWithGenres_df = movies_df.copy()
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())
print(ratings_df.head())

ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
print(inputMovies)

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userMovies)

userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable)
print(inputMovies['rating'])

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
print(userProfile)

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())
print(genreTable.shape)

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())

recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
print(recommendationTable_df.head())
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])

