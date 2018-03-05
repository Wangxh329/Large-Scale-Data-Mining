============================ EE219 Project 4 ==============================
Jui Chang 
Wenyang Zhu 
Xiaohan Wang 
Yang Tang 

======================= Environment & Dependencies ========================
Imported libraries:

import pandas as pd
import numpy as np
import re
import math
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

Functions:

replace_str_with_int(data, column, insert_pos, truncate_pos=0, map_day=None) for all scalar encoding 
1. Input: 
	data: original dataset
	column: columns with string values to be replaced
	insert_pos: insert position of new columns
    truncate_pos: truncated position in string values
2. Output: original data encoded with scalar method

get_sizes(days, data) for Q1
1. Input: 
	days: plotted days period
	data: the processed dataset
2. Output: all backup size data in plotted days period

plot_size_vs_days(period, total_size) for Q1
1. Input: 
	period: plotted days period
	total_size: the backup size data in this period
2. Output: backup size vs days figure

rmse(predictions, targets) for all calculatings of RMSE
1. Input:
    predictions: fitted values
    targets: true values
2. Output: RMSE values

select_features(method, method_str) for Q2-(a)-iii
1. Input:
    method: feature selection method
    method_str: the name of the feature selection method 
2. Output: the 2D array with values of selected features

linear_regression_model(input_arr, method_str) for Q2-(a)-iii
1. Input:
    input_arr: the selected features array after using specific feature selection methods
    method_str: the name of the feature selection method
2. Output: calculate RMSE and plot fitted-true values figure and residual-fitted values figure

one_hot_encoding(input_arr, one_hot_pos) for Q2-(a)-iv
1. Input:
    input_arr: the selected features array after using specific feature selection methods
    one_hot_pos: the positions in the input_arr to be one-hot encoded
2. Output: one-hot encoded input-arr

find_optimal_combination(input_arr, alg, kf_split=10, num_combines=32) for Q2-(a)-v
1. Input:
    input_arr: the selected features array after using specific feature selection methods
    alg: the algorithm used to fit and predict data
    kf_split: number of k-folds
    num_combines: number of combinations
2. Output: 
	best_combination: best combination of whether a feature is one-hot encoded or not
	best_test_rmse: RMSE value of the model with best combination

combinaton_trans(combine) for Q2-(a)-v
1. Input:
    combine: an int representing a kind of combination
2. Output: 
	a string describing which features are one-hot encoded and which are not

Ridge_Lasso_Regularizer(name) for Q2-(a)-v
1. Input:
    name: 'Ridge Regularizer' or 'Lasso Regularizer'
2. Output: 
	print optimal combination, optimal parameters and some other information of input algorithm

FOR Q2_b:
# Ignore warnings:
Ignore warnings raised by oob errors due to few training data

# Load data:
Load data from csv file and translate strings to crossponding category values

# part i:
Build Random Forest (RF) regressor with number of trees = 20, max number of features = 5, max depth of tree = 4, bootstrap = True and oob_score = True

# part ii:
Loop number of trees from 1 to 200 and number of features from 1 to 5, build RFs and calculate OOB errors and test RMSEs
Find optimal number of trees and number of features

# part iii:
Loop number of tree from 1 to 200 with gap = 10 and minimum impurity decrease from 0 to 0.2 with gap 0.04
Find optimal number of trees and minimum impurity decrease

# part iv:
Given optimal hyper-parameters

# part v:
Build RF regressor with optimal hyper-parameters, and plot regression result.


get_x_y(activation) for Q2-(c)
1. Input:
    activation: activation function of current MLPRegressor model
2. Output: 
	a dict in which keys are number of hidden units and values are mean test RMSEs

=============================== Instruction ===============================
Each .py file solves one question and can be run seperately to get results
in console output.
	- network_backup_dataset.csv: the original dataset
	- Q1_load_dataset.py for Question 1.
	- Q2_a_i_scalar_encoding.py for Question 2-a-i
	- Q2_a_ii_data_preprocessing.py for Question 2-a-ii
	- Q2_a_iii_feature_selection.py for Question 2-a-iii
	- Q2_a_iv_feature_encoding.py for Question 2-a-iv
	- Q2_a_v_control_illconditioning_and_overfitting.py for Question 2-a-v
	- Q2_b_random_forest.py for Question 2-b
	- Q2_c_neural_network_regression.py for Question 2-c
	- Q2_d_i_linear_regression_model.py for Question 2-d-i
	- Q2_d_ii_polynomial_function_model.py for Question 2-d-ii
	- Q2_e_knn.py for Question 2-e	

- all .ipynb files can be run directly on Jupyter Notebook and get the 
results below.

============================= Folder Content ==============================
- codes
	- Q1_load_dataset.py
	- Q2_a_i_scalar_encoding.py
	- Q2_a_ii_data_preprocessing.py
	- Q2_a_iii_feature_selection.py
	- Q2_a_iv_feature_encoding.py
	- Q2_a_v_control_illconditioning_and_overfitting.py
	- Q2_b_random_forest.py
	- Q2_c_neural_network_regression.py
	- Q2_d_i_linear_regression_model.py
	- Q2_d_ii_polynomial_function_model.py
	- Q2_e_knn.py

	- Q1_load_dataset.py.ipynb
	- Q2_a_i_scalar_encoding.ipynb
	- Q2_a_ii_data_preprocessing.ipynb
	- Q2_a_iii_feature_selection.ipynb
	- Q2_a_iv_feature_encoding.ipynb
	- Q2_a_v_control_illconditioning_and_overfitting.ipynb
	- Q2_c_neural_network_regression.ipynb
	- Q2_d_i_linear_regression_model.ipynb
	- Q2_d_ii_polynomial_function_model.ipynb
	- network_backup_dataset.csv

- readme.txt
- report.pdf

