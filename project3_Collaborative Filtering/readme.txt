============================ EE219 Project 3 ==============================
Jui Chang 
Wenyang Zhu 
Xiaohan Wang 
Yang Tang 

======================= Environment & Dependencies ========================
Imported libraries:

import pandas as pd
from collections import defaultdict
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import accuracy
from surprise.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


Functions:

pop_set_trim(testset) for Q12
1. Input: data to be trimmed, in this question, it's test set
2. Output: data(movie) with rating frequency > 2

unpop_set_trim(testset) for Q13
1. Input: data to be trimmed, in this question, it's test set
2. Output: data(movie) with rating frequency <= 2

highVar_set_trim(testset) for Q14
1. Input: data to be trimmed, in this question, it's test set
2. Output: data(movie) with rating frequency >= 5 and rating variance >=2

nmf(dataName, data, biased=True) for Q17-21,24-28
1. Input:
    dataName: name of the dataset, including 'all movies', 'popular movies', 'unpopular movies' and 'high variance movies'
    data: train data to be trained by NNMF
    biased: if True, bias is used; otherwise, it is not used in NNMF model
2. Output:
    Graphs of 'mae vs k' and 'rmse vs k'

drawRoc(model, i, k) for Q22,29
1. Input:
    model: NNMF model with n_factors = optimal K and biased = False in Q22 but True in Q29
    i: the index of the chosen threshold, e.g., when i = 0, we choose threshold = 2.5
    k: optimal K used for model
2. Output:
    Graphs of ROC curve of NNMF model with different thresholds

cross_validation(dataset) for Q30-33
1. Input:
    dataName: name of the dataset, including "MovieLens Dataset", "Popular Movies Dataset", "Unpopular Movies Dataset", "High Variance Movies Dataset"
2. Output:
    print values of RMSEs

precision_recall_at_t(predictions, t, threshold=3) for Q36-39
1. Input:
    predictions: result of a filter's test() function
    t: length of recommendation list
    threshold: threshold of deciding whether a user likes an item or not, default is 3
2. Output:
    a tuple of dict, (precisions, recalls) which stores each prediction's precision and recall

cal_roc(model) for Q34
1. Input:
    model: a collaborative filter
2. Output:
    fpr, tpr and roc_auc scores

draw_t_prec_recall(algo, kf, t_low, t_high, thre) for Q36-39
1. Input:
    algo: a collaborative filter
    t: length of recommendation list
    t_low: lower bound of t
    t_high: higher bound of t
    thre: threshold of deciding whether a user likes an item or not
2. Output:
    a tuple of list, (ts, precision, recall).
    ts contains of t values, precision and recall contains all corresponding precision and recall values

testset_trim(testset, t, threshold=3) for Q36-39
1. Input:
    testset: a testset
    t: length of recommendation list
    threshold: threshold of deciding whether a user likes an item or not, default is 3
2. Output:
    trimmed testset

=============================== Instruction ===============================
Each .py file solves one question and can be run seperately to get results
in console output.
- 3_MovieLens_dataset_Q1_9.py: for Question 1 to 9.

- 4_neighborhood_based_collaborative_filter_Q10_15.py: for Question 10 to 15.

- 5_attach_preprocessing.py: for the data preprocessing of part 5.

- 5_model_based_collaborative_filter_Q16_29.py: for Question 16 to 29.

- 6_naive_collaborative_filter_Q30_33.py: for Question 30 to 33.

- 7_performance_comparison_Q34.py: for Question 34.

- 8_ranking_Q35_39.py: for Question 35 to 39.

- movies.csv/ratings.csv: the original data

- bin2.5.csv/bin3.csv/bin3.5.csv/bin4.csv/popular.csv/unpopular.csv/variance.csv: the preprocessed data

- all .ipynb files can be run directly on Jupyter Notebook and get the 
results below.

============================= Folder Content ==============================
- codes
    - 3_MovieLens_dataset_Q1_9.py
    - 4_neighborhood_based_collaborative_filter_Q10_15.py
    - 5_attach_preprocessing.py
    - 5_model_based_collaborative_filter_Q16_29.py
    - 6_naive_collaborative_filter_Q30_33.py
    - 7_performance_comparison_Q34.py
    - 8_ranking_Q35_39.py
    - 3_MovieLens_dataset_Q1_9.ipynb
    - 4_neighborhood_based_collaborative_filter_Q10_15.ipynb
    - 5_attach_preprocessing.ipynb
    - 5_model_based_collaborative_filter_Q16_29.ipynb
    - 6_naive_collaborative_filter_Q30_33.ipynb
    - 7_performance_comparison_Q34.ipynb
    - 8_ranking_Q35_39.ipynb
    - movies.csv/ratings.csv
    - bin2.5.csv/bin3.csv/bin3.5.csv/bin4.csv
    - popular.csv/unpopular.csv/variance.csv

- readme.txt
- report.pdf

