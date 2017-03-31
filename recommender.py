# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:14:15 2017

@author: shubh
"""

import pandas as pd

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/Users/shubh/Documents/More Learning/Projects/Recommender System/Movielens-dataset/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('C:/Users/shubh/Documents/More Learning/Projects/Recommender System/Movielens-dataset/ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('C:/Users/shubh/Documents/More Learning/Projects/Recommender System/Movielens-dataset/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

print users.shape
users.head()

print ratings.shape
ratings.head()

print items.shape
items.head()

#Splitting the ratings data in training and test data.
# problem giver has pre-splitted the data. So, just loading the already split data

ratings_base = pd.read_csv('C:/Users/shubh/Documents/More Learning/Projects/Recommender System/Movielens-dataset/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('C:/Users/shubh/Documents/More Learning/Projects/Recommender System/Movielens-dataset/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1') 

#checking rows and columns
ratings_base.shape

ratings_test.shape

#using graphlab, so need to convert them to S-frames.
import graphlab

train_data = graphlab.SFrame(ratings_base)

test_data = graphlab.SFrame(ratings_test)

#popularity model

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id',item_id='movie_id', target='rating')
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)

#collabrative filtering
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)

item_sim_recomm.print_rows(num_rows=25)

#comparing
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])

graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])
