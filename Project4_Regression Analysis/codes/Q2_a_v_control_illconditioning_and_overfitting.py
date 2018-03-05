
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
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


# load data from file
data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week_orig', 'start_time','work_flow_orig','file_name_orig','size','duration']

def replace_str_with_int(data, column, insert_pos, truncate_pos=0, map_day=None):
    new_col = []
    for item in data[column]:
        if map_day:
            new_col.append(map_day[item])
        else:
            new_col.append(int(item[truncate_pos:]))
    
    data.insert(insert_pos, column[:len(column) - 5], new_col)
    data.drop(column, 1, inplace = True)       

# 1 encode day of week
map_day = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
replace_str_with_int(data, 'day_of_week_orig', 2, 0, map_day)

# 2 encode work flow
replace_str_with_int(data, 'work_flow_orig', 3, 10)

# 3 encode file name
replace_str_with_int(data, 'file_name_orig', 4, 5)

# extract input and output
input_arr = []
for row in range(len(data)):
    input_arr.append(data.loc[row, 'week':'file_name'].values)

output_arr = data.loc[:, 'size'].values

def one_hot_encoding(input_arr, one_hot_pos):
    enc = OneHotEncoder(n_values='auto', categorical_features=one_hot_pos, 
                        sparse=False, handle_unknown='error')
    return enc.fit_transform(input_arr)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def find_optimal_combination(input_arr, alg, kf_split=10, num_combines=32):
    # return values of this function
    best_test_rmse = float("inf")
    best_combination = 0
    
    kf = KFold(n_splits=kf_split, shuffle=False)
    for combine in range(num_combines):
        # 00000~11111 (0~31) using bit operation
        one_hot_pos = []
        for pos in range(5):
            if ((combine >> pos) & 1) == 1:
                one_hot_pos.append(pos)
        tranformed_input = one_hot_encoding(input_arr, one_hot_pos)

        test_rmses = []
        
        for train_index, test_index in kf.split(tranformed_input):
            
            train_in = [tranformed_input[i] for i in train_index]
            train_out = [output_arr[i] for i in train_index]
            test_in = [tranformed_input[i] for i in test_index]
            test_out = [output_arr[i] for i in test_index]

            alg.fit(train_in, train_out)
            test_pre = alg.predict(test_in)

            test_rmses.append(rmse(test_pre, test_out))

        mean_test_rmse = np.mean(test_rmses)
        
        if mean_test_rmse < best_test_rmse:
            best_test_rmse = mean_test_rmse
            best_combination = combine

    return best_combination, best_test_rmse

def combinaton_trans(combine):
    variables = ['week','day_of_week','start_time','work_flow','file_name']
    use_one_hot = []
    use_scalar = []
    for pos in range(5):
            if ((combine >> pos) & 1) == 1:
                use_one_hot.append(variables[pos])
            else:
                use_scalar.append(variables[pos])
    return 'use_one_hot: ' + str(use_one_hot) + '\nuse_scalar: ' + str(use_scalar)

# Ridge Regularizer & Lasso Regularizer
def Ridge_Lasso_Regularizer(name):
    print('using ' + name)
    best_combine = None
    lowest_rmse = float("inf")
    optimal_alpha = 1
    for alpha in [0.001, 0.01, .1, .5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        reg = None
        if name == 'Ridge Regularizer':
            reg = linear_model.Ridge (alpha = alpha)
        else:
            reg = linear_model.Lasso(alpha=alpha)
        best_combination, best_test_rmse = find_optimal_combination(input_arr, reg, kf_split=10, num_combines=32)
        if best_test_rmse < lowest_rmse:
            best_combine = best_combination
            lowest_rmse = best_test_rmse
            optimal_alpha = alpha

    print ('Optimal Combination:')
    print (combinaton_trans(best_combine)) 
    print ('Optimal Alpha: ' + str(optimal_alpha))
    print ('Optimal Test Rmse: ' + str(lowest_rmse))

    one_hot_pos = []
    for pos in range(5):
        if ((best_combine >> pos) & 1) == 1:
            one_hot_pos.append(pos)
    tranformed_input = one_hot_encoding(input_arr, one_hot_pos)

    alg = linear_model.Ridge (alpha = optimal_alpha)
    alg.fit(tranformed_input, output_arr)
    print ('estimated coefficients: ')
    print (str(alg.coef_))
    print ('-----------------------------------------')

Ridge_Lasso_Regularizer('Ridge Regularizer')
Ridge_Lasso_Regularizer('Lasso Regularizer')

# Elastic Net Regularizer:
print('using ' + 'Elastic Net Regularizer')
best_combine = None
lowest_rmse = float("inf")
optimal_alpha = .2
optimal_l1_ratio = 0.001
for alpha in [0.01, 0.1, 1, 10, 100]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        best_combination, best_test_rmse = find_optimal_combination(input_arr, reg, kf_split=10, num_combines=32)
        if best_test_rmse < lowest_rmse:
            best_combine = best_combination
            lowest_rmse = best_test_rmse
            optimal_alpha = alpha

print ('Optimal Combination:')
print (combinaton_trans(best_combine)) 
print ('Optimal Alpha1: ' + str(optimal_alpha * optimal_l1_ratio))
print ('Optimal Alpha2: ' + str(optimal_alpha * (1-optimal_l1_ratio)))
print ('Optimal Test Rmse: ' + str(lowest_rmse))

one_hot_pos = []
for pos in range(5):
    if ((best_combine >> pos) & 1) == 1:
        one_hot_pos.append(pos)
tranformed_input = one_hot_encoding(input_arr, one_hot_pos)

alg = ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1_ratio)
alg.fit(tranformed_input, output_arr)
print ('estimated coefficients: ')
print (str(alg.coef_))
print ('-----------------------------------------')

# using un-regularized best model
print('using un-regularized best model')

best_combine = 22
one_hot_pos = []
for pos in range(5):
    if ((best_combine >> pos) & 1) == 1:
        one_hot_pos.append(pos)
tranformed_input = one_hot_encoding(input_arr, one_hot_pos)

lr = linear_model.LinearRegression()
lr.fit(tranformed_input, output_arr)

print ('Optimal Combination:')
print (combinaton_trans(best_combine))
print ('Optimal Test Rmse: 0.0883701294703')
print ('estimated coefficients: ')
print (str(lr.coef_))
print ('-----------------------------------------')

