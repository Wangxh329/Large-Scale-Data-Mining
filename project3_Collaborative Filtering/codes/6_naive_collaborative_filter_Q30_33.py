
# coding: utf-8

# In[6]:


import pandas as pd
import os
import numpy as np
import pandas as pd
import random
from surprise import AlgoBase
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


# naive collaborative filter
class NaiveCollaborativeFilter(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute rating means of each user
        
        self.trainset = trainset
        self.the_means = {}
        for key in self.trainset.ur:
            urs = self.trainset.ur[key]
            mean = np.mean([r for (_, r) in urs])
            self.the_means[key] = mean

        return self

    def estimate(self, u, i):
        if self.the_means.__contains__(u):
            return self.the_means[u]
        else:
            return 3


# calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# read csv file
file_path = os.path.expanduser('ratings.csv')
df = pd.read_csv(file_path)
del df['timestamp']

# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# 10 fold cross validation
def cross_validation(dataset):
    testlen = int(len(df) / 10)
    total_rmse = 0
    print(dataset + ':')
    for i in range(10):
        # split data into 10 pieces
        sp = int(len(df) / 10 * i)
        df1 = df[0:sp]
        df2 = df[sp:sp+testlen]
        df3 = df[sp+testlen:]

        traindf = pd.concat([df1,df3],ignore_index=True)
        reader1 = Reader(rating_scale=(0, 5))
        trainset = Dataset.load_from_df(traindf[['userId', 'movieId', 'rating']], reader1)

        alg = NaiveCollaborativeFilter()
        alg.fit(trainset.build_full_trainset())

        if dataset == "Popular Movies Dataset":
            df2['size'] = df2.groupby(['movieId']).movieId.transform(np.size)
            df2 = df2[df2['size'] > 2]

        if dataset == "Unpopular Movies Dataset":
            df2['size'] = df2.groupby(['movieId']).movieId.transform(np.size)
            df2 = df2[df2['size'] <= 2]

        if dataset == "High Variance Movies Dataset":
            df2['size'] = df2.groupby(['movieId']).movieId.transform(np.size)
            df2 = df2[df2['size'] >= 5]
            df2['var'] = df2['rating'].groupby(df2['movieId']).transform(lambda arr:np.mean((arr - arr.mean()) ** 2))
            df2 = df2[df2['var'] >= 2]
    
        reader2 = Reader(rating_scale=(0, 5))
        testset = Dataset.load_from_df(df2[['userId', 'movieId', 'rating']], reader2)
        testset = [(u, i, r) for (u, i, r) in testset.build_full_trainset().all_ratings()]

        prediction = alg.test(testset)
    
        real = []
        est = []

        for j in range(len(prediction)):
            if not prediction[j][4]['was_impossible']:
                real.append(prediction[j][2])
                est.append(prediction[j][3])
    
        cur_rmse = rmse(np.array(real), np.array(est))
        total_rmse += cur_rmse
        print(str(i) + 'th rmse: ' + str(cur_rmse))

    final_rmse = total_rmse / 10
    print('final_rmse: ' + str(final_rmse))
    print('')


# Q30
cross_validation("MovieLens Dataset")

# Q31
cross_validation("Popular Movies Dataset")

# Q32
cross_validation("Unpopular Movies Dataset")

# Q33
cross_validation("High Variance Movies Dataset")


