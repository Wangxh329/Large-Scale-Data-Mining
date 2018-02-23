
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict


file_path = os.path.expanduser('ratings.csv')
dataset = pd.read_csv(file_path, delimiter=',')
# dataset.iloc[0,1]
dat_mat = dataset.copy()

# R data frame
R_df = dat_mat.pivot_table('rating', 'userId', 'movieId', fill_value=0)
# R matrix
R = pd.DataFrame.as_matrix(R_df)

# column name(movie ID), rows(User ID)
movieId_uni = list(R_df)
userId_uni = list(R_df.index)

# Q1
# total available rating = 10004, total possible rating = R(len) * R(width)
sparsity = round(1 - len(dat_mat) / np.prod(R.shape), 4)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

# Q2
bar_add = np.zeros((1,10))
for i in np.arange(R.shape[0]):
    bar = plt.hist(R[i,:], bins=[0.01, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],ec='black')[0]
    bar_add += bar
    print('i = %d' %i)
pass
plt.plot(np.array([[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]]), bar_add)
rating_all = [0.8] * int(bar_add[0,1]) + [1.3] * int(bar_add[0,2]) +  [1.8]* int(bar_add[0,3]) +[2.3] * int(bar_add[0,4])+[2.8] * int(bar_add[0,5])+  [3.3]* int(bar_add[0,6])+ [3.8]* int(bar_add[0,7])+[4.3] * int(bar_add[0,8])+  [4.8]* int(bar_add[0,9])
plt.hist(rating_all, bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], ec='black')
plt.title("Movie rating ")
plt.xlabel("Rating of movie")
plt.ylabel("Count of Movie")
plt.show()

# Q3
dist_movie = np.sum(R!=0, axis=0)
dist_movie_copy = dist_movie.copy()

dist_movie_sort = sorted(dist_movie_copy)[::-1]
dist_movie_idx = np.argsort(dist_movie_copy)[::-1]
real_movie_idx = []

for i in range(len(dist_movie_idx)):
    real_movie_idx.append(movieId_uni[dist_movie_idx[i]])
pass
dist_movie_idx_str= []
for i in range(len(dist_movie_idx)):
    dist_movie_idx_str.append(str(real_movie_idx[i]))
pass
plt.plot(dist_movie_sort)
plt.title("Distribution of rating among movie")
plt.xlabel("Movie Index")
plt.ylabel("Number of ratings Received")
plt.show()

# Q4
dist_user = np.sum(R!=0, axis=1)
dist_user_copy = dist_user.copy()
dist_user_sort = sorted(dist_user_copy)[::-1]
dist_user_idx = np.argsort(dist_user_copy)[::-1]
real_user_idx = []
for i in range(len(dist_user_idx)):
    real_user_idx.append(userId_uni[dist_user_idx[i]])
pass
plt.plot(dist_user_sort)
plt.title("Distribution of rating among user")
plt.xlabel("User Index")
plt.ylabel("Number of ratings Received")
plt.show()

# Q6
R_df_na = dat_mat.pivot_table('rating', 'userId', 'movieId') # no rating as nan
var_movie = np.var(R_df_na, axis=0)
plt.hist(var_movie, bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], ec='black')
plt.title("Movie rating variance")
plt.xlabel("Rating Variance")
plt.ylabel("Count of Movie")
plt.show()

