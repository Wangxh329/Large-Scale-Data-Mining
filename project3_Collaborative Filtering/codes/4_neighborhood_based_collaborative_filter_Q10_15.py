
# coding: utf-8

# In[8]:


from surprise.model_selection.validation import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


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

#================== functions #=====================
# preprocess
def pop_set_trim(testset):
    # find the pop index
    freq_movie = np.sum(R!=0, axis=0)
    pop_idx_R = list(np.where(freq_movie > 2))[0]
    pop_index=[]
    for i in range(len(pop_idx_R)):
        pop_index.append(movieId_uni[pop_idx_R[i]])
    pass

    # extract the popular testset
    pop_trim_test = []
    for i in range(len(testset)):
        if int(testset[i][1]) in pop_index:
            pop_trim_test.append(testset[i])
        pass
    pass
    return pop_trim_test

def unpop_set_trim(testset):
    # find the pop index
    freq_movie = np.sum(R!=0, axis=0)
    unpop_idx_R = list(np.where(freq_movie <= 2))[0]
    unpop_index=[]
    for i in range(len(unpop_idx_R)):
        unpop_index.append(movieId_uni[unpop_idx_R[i]])
    pass
    # extract the unpopular testset
    unpop_trim_test = []
    for i in range(len(testset)):
        if int(testset[i][1]) in unpop_index:
            unpop_trim_test.append(testset[i])
        pass
    pass
    return unpop_trim_test

freq_movie = np.sum(R!=0, axis=0)
R_df_na = dat_mat.pivot_table('rating', 'userId', 'movieId') # no rating as nan
var_movie = np.var(R_df_na, axis=0)
def highVar_set_trim(testset):
    pop_idx_R = list(np.where(freq_movie >= 5))[0]  # at least 5 ratings
    high_var_index = []
    high_var_trim_test = []
    for i in range(len(pop_idx_R)):
        # variance at least 2
        if list(var_movie)[pop_idx_R[i]] >= 2:
            high_var_index.append(movieId_uni[pop_idx_R[i]])
        pass
    pass
    for i in range(len(testset)):
        if int(testset[i][1]) in high_var_index:
            high_var_trim_test.append(testset[i])
        pass
    pass
    return high_var_trim_test

# Q10
# In order to fit surprise
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(file_path, reader=reader)

acc_cv = np.zeros((2,50))
sim_options = {'name': 'pearson'}
i=0
for k in range(2,101,2):
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    cv1 = cross_validate(algo, data, measures=['RMSE', 'MAE'],
                         cv=10, verbose=False)
    acc_cv[0,i] = np.mean(cv1['test_rmse'])
    acc_cv[1,i] = np.mean(cv1['test_mae'])
    print('test_rmse = %f, test_mae = %f' %(acc_cv[0, i], acc_cv[1, i]))
    i = i+1
pass
ks = np.arange(2,101,2)

plt.xlabel('k')
plt.ylabel('Error value')
plt.title('Test RMSE and MAE vs k in KNN with 10 Validation')
plt.plot(ks, acc_cv[0,:])
plt.plot(ks, acc_cv[1,:])
plt.legend(['RMSE', 'MAE'], loc='upper right')
plt.show()

# Q11
# minimum k = 24, RSME = 0.9201 MAE = 0.7011

# Q12
# define a cross-validation iterator
kf = KFold(n_splits=10)
rsme_pop = []
for k in range(2,101,2):
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    rsme_pre = []
    for trainset, testset in kf.split(data):
        # train and test algorithm
        algo.fit(trainset)
        pop_trim_test = pop_set_trim(testset=testset)
        predictions = algo.test(pop_trim_test)
        # Compute and print Root Mean Squared Error
        rsme_pre.append(accuracy.rmse(predictions, verbose=False))
    pass
    rsme_pop.append(np.mean(rsme_pre))
pass
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('Popular Movie Trimming :RMSE vs k in KNN')
plt.plot(ks, rsme_pop)
plt.show()
# k = 26, RSME = 0.901

# Q13
rsme_unpop = []
for k in range(2,101,2):
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    rsme_pre = []
    for trainset, testset in kf.split(data):
        # train and test algorithm
        algo.fit(trainset)
        unpop_trim_test = unpop_set_trim(testset=testset)
        predictions = algo.test(unpop_trim_test)
        # Compute and print Root Mean Squared Error
        rsme_pre.append(accuracy.rmse(predictions, verbose=False))
    pass
    rsme_unpop.append(np.mean(rsme_pre))
pass
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('Unpopular Movie Trimming :RMSE vs k in KNN')
plt.plot(ks, rsme_unpop)
plt.ylim((1.1,1.3))
plt.show()
# RMSE not decrease as k increase, RMSE stays 1.190
# test set is small so RMSE unstable

# Q14
rmse_highVar = []
for k in range(2,101,2):
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    rsme_pre = []
    for trainset, testset in kf.split(data):
        # train and test algorithm
        algo.fit(trainset)
        highV_trim_test = highVar_set_trim(testset=testset)
        predictions = algo.test(highV_trim_test)
        # Compute and print Root Mean Squared Error
        rsme_pre.append(accuracy.rmse(predictions, verbose=False))
    pass
    rmse_highVar.append(np.mean(rsme_pre))
pass
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('High Variance Movie Trimming :RMSE vs k in KNN')
plt.plot(ks, rmse_highVar)
plt.ylim((1.45,1.8))
plt.show()
# At first k from 2 to 6, RMSE drops,
# from k>6, RMSE starts to be unstable, RMSE value changes a lot as k increases
# k=6, RMSE = 1.647
# test set is small, so RMSE unstable
# Variance is high, so the RMSE stays large

# Q15 ROC curve
algo = KNNWithMeans(k=24, sim_options=sim_options)
threshold = 4
from surprise.model_selection import train_test_split
# train and test algorithm
trainset, testset = train_test_split(data, test_size=0.1)
algo.fit(trainset)
predictions = algo.test(testset)

real_y = []
est_y = []
for i in range(len(predictions)):
    est_y.append(predictions[i].est)
    if testset[i][2] >= threshold:
        real_y.append(1.0)
    else:
        real_y.append(0.0)
    pass
pass

from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(real_y, est_y)
AUC = roc_auc_score(real_y, est_y)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve with threshold = %f, AUC score = %.3f' %(threshold, AUC))
plt.show()
# when ROC curve with threshold=3.5, the AUC value are the largest,
# which means it gives us the most true positive rate and less False negative rate
# when threshold = 3.5, the threshold value gives us the optimal result because


