import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    data.drop(column, 1, inplace=True)

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


# standardization
scaler = StandardScaler()
input_arr = scaler.fit_transform(input_arr)
#output_arr = scaler.fit_transform(output_arr.reshape(-1, 1))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# create an array store all rmse
# #1st row: k rmse_matrix[0,:]
# #2nd row: test rmse rmse_matrix[1,:]
# #3rd row: train rmse rmse_matrix[2,:]
rmse_matrix = np.zeros((3,30))
rmse_matrix[0,:] = np.arange(1,31)
input_arr_small = input_arr
output_arr_small = output_arr.reshape(output_arr.shape[0],)

for k in range(1,31):#range(24,26,2):
    neigh = KNeighborsRegressor(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=False)
    rmse_test = []
    rmse_train = []
    y_test_predict_history= np.array([])
    y_train_predict_history = np.array([])

    for train_index, test_index in kf.split(input_arr_small):

        X_train, X_test = input_arr_small[train_index], input_arr_small[test_index]
        y_train, y_test = output_arr_small[train_index], output_arr_small[test_index]
        neigh.fit(X_train, y_train)

        y_test_predict = np.zeros_like(y_test)
        y_train_predict = np.zeros_like(y_train)
        #do prediction
        for j1 in range(y_test.shape[0]):
            y_test_predict[j1] = neigh.predict([X_test[j1]])
        pass
        for i1 in range(y_train.shape[0]):
            y_train_predict[i1] = neigh.predict([X_train[i1]])
        pass
        #record prediction data and train data
        y_test_predict_history = np.append(y_test_predict_history, y_test_predict)
        y_train_predict_history = np.append(y_train_predict_history, y_train_predict)
        rmse_test.append(rmse(y_test_predict, y_test))
        rmse_train.append(rmse(y_train_predict, y_train))
    pass
    print(k)
    rmse_matrix[1, (k-1)] = np.mean(rmse_test)
    rmse_matrix[2, (k-1)] = np.mean(rmse_train)
pass
#plot train and test RMSE vs k
plt.plot(rmse_matrix[0,:k-1], rmse_matrix[1,:k-1])
plt.plot(rmse_matrix[0,:k-1], rmse_matrix[2,:k-1])
plt.ylabel('RMSE')
plt.xlabel('k')
plt.title('KNN Regression: Train & Test RMSE vs k')
plt.legend(['Test ', 'Train'])
plt.show()

# min RMSE at k=5
k = 5
neigh = KNeighborsRegressor(n_neighbors=k)
kf = KFold(n_splits=10, shuffle=False)

y_test_predict_history = np.array([])
y_train_predict_history = np.array([])
y_test_truth_history = np.array([])
rmse_cv_train=[]
rmse_cv_test=[]
for train_index, test_index in kf.split(input_arr_small):

    X_train, X_test = input_arr_small[train_index], input_arr_small[test_index]
    y_train, y_test = output_arr_small[train_index], output_arr_small[test_index]
    neigh.fit(X_train, y_train)

    y_test_truth_history = np.append(y_test_truth_history, y_test)

    y_test_predict = np.zeros_like(y_test)
    y_train_predict = np.zeros_like(y_train)
    # do prediction
    for j1 in range(y_test.shape[0]):
        y_test_predict[j1] = neigh.predict([X_test[j1]])
    pass
    for i1 in range(y_train.shape[0]):
        y_train_predict[i1] = neigh.predict([X_train[i1]])
    pass
    # record prediction data and train data
    y_test_predict_history = np.append(y_test_predict_history, y_test_predict)
    y_train_predict_history = np.append(y_train_predict_history, y_train_predict)
    rmse_cv_test.append(rmse(y_test_predict, y_test))
    rmse_cv_train.append(rmse(y_train_predict, y_train))
pass
rmse_cv = np.array([rmse_cv_train,rmse_cv_test])
rmse_cv1 = np.transpose(rmse_cv)

# plot fitted values vs true values
plt.scatter(y_test_truth_history, y_test_predict_history, color='deeppink', edgecolors='k')
plt.plot([y_test_truth_history.min(), y_test_truth_history.max()], [y_test_predict_history.min(), y_test_predict_history.max()], 'k--', lw=2)
plt.ylabel('Fitted Backup Size (GB)')
plt.xlabel('True Backup Size (GB)')
plt.title('Fitted Values vs True Values: KNN regression')
plt.show()

# plot residuals vs fitted values
residuals = np.subtract(y_test_truth_history, y_test_predict_history)
plt.scatter(y_test_predict_history, residuals, color='deeppink', edgecolors='k')
plt.plot([y_test_predict_history.min(), y_test_predict_history.max()], [0, 0], 'k--', lw=2)
plt.ylabel('Residuals')
plt.xlabel('Fitted Backup Size (GB)')
plt.title('Residuals vs Fitted Values: KNN regression')
plt.show()

