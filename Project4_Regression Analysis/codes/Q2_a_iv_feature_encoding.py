
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder

# load data from file
data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week_orig', 'start_time','work_flow_orig','file_name_orig','size','duration']

# 1 use scalar encoding method
def replace_str_with_int(data, column, insert_pos, truncate_pos=0, map_day=None):
    new_col = []
    for item in data[column]:
        if map_day:
            new_col.append(map_day[item])
        else:
            new_col.append(int(item[truncate_pos:]))
    
    data.insert(insert_pos, column[:len(column) - 5], new_col)
    data.drop(column, 1, inplace = True)       

# 1) encode day of week
map_day = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
replace_str_with_int(data, 'day_of_week_orig', 2, 0, map_day)

# 2) encode work flow
replace_str_with_int(data, 'work_flow_orig', 3, 10)

# 3) encode file name
replace_str_with_int(data, 'file_name_orig', 4, 5)

# extract input and output
input_arr = []
for row in range(len(data)):
    input_arr.append(data.loc[row, 'week':'file_name'].values)

output_arr = data.loc[:, 'size'].values

# 2 use one-hot encoding method
def one_hot_encoding(input_arr, one_hot_pos):
    enc = OneHotEncoder(n_values='auto', categorical_features=one_hot_pos, 
                        sparse=False, handle_unknown='error')
    return enc.fit_transform(input_arr)

# calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

lr = linear_model.LinearRegression()
kf = KFold(n_splits=10, shuffle=False)

train_rmses_means = []
test_rmses_means = []
combines = [i + 1 for i in range(32)]

for combine in range(32):
    # 00000~11111 (0~31) using bit operation
    one_hot_pos = []
    for pos in range(5):
        if ((combine >> pos) & 1) == 1:
            one_hot_pos.append(pos)
    tranformed_input = one_hot_encoding(input_arr, one_hot_pos)
    
    train_rmses = []
    test_rmses = []

    for train_index, test_index in kf.split(tranformed_input):
        train_in = [tranformed_input[i] for i in train_index]
        train_out = [output_arr[i] for i in train_index]
        test_in = [tranformed_input[i] for i in test_index]
        test_out = [output_arr[i] for i in test_index]

        lr.fit(train_in, train_out)

        test_pre = lr.predict(test_in)
        train_pre = lr.predict(train_in)

        train_rmse = rmse(train_pre, train_out)
        test_rmse = rmse(test_pre, test_out)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    train_rmses_means.append(np.mean(train_rmses))
    test_rmses_means.append(np.mean(test_rmses))

df = pd.DataFrame({
    'combine' : combines,
    'mean_train_rmse' : train_rmses_means,
    'mean_test_rmse' : test_rmses_means
}, columns = ['combine', 'mean_train_rmse', 'mean_test_rmse'])

display(df)
train_rmses = df['mean_train_rmse'].values
test_rmses = df['mean_test_rmse'].values

# find combination with best performance
best_combine = None
second_best_combine = None
third_best_combine = None
best_mean_rmses = float("inf") 
second_best_mean_rmses = float("inf")
third_best_mean_rmses = float("inf")
for i in range(len(train_rmses)):
    if (train_rmses[i]+test_rmses[i]<best_mean_rmses):
        third_best_mean_rmses = second_best_mean_rmses
        third_best_combine = second_best_combine
        second_best_mean_rmses = best_mean_rmses
        second_best_combine = best_combine
        best_mean_rmses = train_rmses[i] + test_rmses[i]
        best_combine = i
    elif (train_rmses[i]+test_rmses[i]<second_best_mean_rmses):
        third_best_mean_rmses = second_best_mean_rmses
        third_best_combine = second_best_combine
        second_best_mean_rmses = train_rmses[i] + test_rmses[i]
        second_best_combine = i
    elif (train_rmses[i]+test_rmses[i]<third_best_mean_rmses):
        third_best_mean_rmses = train_rmses[i] + test_rmses[i]
        third_best_combine = i

print ('The best combination is: '+str(best_combine))
print ('The best mean_train_rmse is: '+str(train_rmses[best_comine])+', and the best mean_test_rmse is: '+str(test_rmses[best_combine]))
print ('The second best combination is: '+str(second_best_combine))
print ('The second best mean_train_rmse is: '+str(train_rmses[second_best_combine])+', and the second best mean_test_rmse is: '+str(test_rmses[second_best_combine]))
print ('The third best combination is: '+str(third_best_combine))
print ('The third best mean_train_rmse is: '+str(train_rmses[third_best_combine])+', and the third best mean_test_rmse is: '+str(test_rmses[third_best_combine]))

# plot training RMSE and test RMSE
plt.figure(figsize=(15,9))
plt.plot(range(1,33), train_rmses, 'deeppink', label='mean train RMSE')
plt.plot(range(1,33), test_rmses, 'deepskyblue', label='mean test RMSE')
plt.ylabel('Encoding Combinations', fontsize = 18)
plt.xlabel('RMSE', fontsize = 18)
plt.title('Training & Test RMSE vs Encoding Combinations', fontsize = 23)
plt.legend(loc='upper left')
plt.show()

