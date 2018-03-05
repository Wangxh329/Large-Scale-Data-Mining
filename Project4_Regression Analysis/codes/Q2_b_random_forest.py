import re
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
import warnings
from sklearn.tree import export_graphviz

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load data
data = pd.read_csv('network_backup_dataset.csv').values
X, y = data[:, :-2], data[:, -2]
days = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
data[:, 1] = [days[d] for d in data[:, 1]]
data[:, 3] = [int(re.sub(r'work_flow_', '', wf)) for wf in data[:, 3]]
data[:, 4] = [int(re.sub(r'File_', '', f)) for f in data[:, 4]]
print('Number of rows of data: ' + str(data.shape[0]))
print('-------------------------------------')

# i
print('Regressor Info:\nNumber of trees: 20\nDepth of each tree: 4\nBootstrap: True\nMaximum number of features: 5\n')
rf = RandomForestRegressor(n_estimators=20, max_features=5, max_depth=4, bootstrap=True, oob_score=True)
scoring = {'mse':make_scorer(mean_squared_error)}
cv = cross_validate(rf, X, y, cv=10, return_train_score=True, scoring=scoring)
print('Average Training RMSE: ' + str(math.sqrt(cv['train_mse'].mean())))
print('Average Test RMSE: ' + str(math.sqrt(cv['test_mse'].mean())))
rf.fit(X, y)
y_pred = rf.predict(X)
residual = y - y_pred
print('Out of Bag error: ' + str(rf.oob_score_))
train_rmse = np.sqrt(cv['train_mse'])
test_rmse = np.sqrt(cv['test_mse'])
print('Training RMSE for 10 folds:')
print(train_rmse)
print('Test RMSE for 10 folds:')
print(test_rmse)
print('-------------------------------------')

plt.figure()
plt.scatter(y, y_pred, c='r', s=6)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')
plt.title('Fitted values vs. True values')
plt.figure()
plt.scatter(y_pred, residual, c='y', s=6)
plt.plot([0, 1], [0, 0], color='navy', lw=2, linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. fitted values')

# ii
numTrees = 200
#gap = 10
numFeatures = 5
oobErrors = np.zeros((numFeatures, numTrees))
testRMSEs = np.zeros((numFeatures, numTrees))
for t in range(1, numTrees+1):
    print('Tree ' + str(t) + ':')
    for f in range(1, numFeatures+1):
        print(f)
        rf = RandomForestRegressor(n_estimators=t, max_features=f, max_depth=4, bootstrap=True, oob_score=True)
        cv = cross_validate(rf, X, y, cv=10, return_train_score=True, scoring=scoring)
        testRMSEs[f-1][t-1] = math.sqrt(cv['test_mse'].mean())
        rf.fit(X, y)
        oobErrors[f-1][t-1] = rf.oob_score_
moob = np.argmax(oobErrors)
roob, coob = moob//numTrees+1, moob%numTrees+1
mrmse = np.argmin(testRMSEs)
rrmse, crmse = mrmse//numTrees+1, mrmse%numTrees+1
print('Maximum OOB errors: ' + str(oobErrors[roob-1][coob-1]) + ' tree ' + str(coob) + ' feature ' + str(roob))
print('Minimum test RMSE: ' + str(testRMSEs[rrmse-1][crmse-1]) + ' tree ' + str(crmse) + ' feature ' + str(rrmse))

x = range(1, numTrees+1)
colors = 'bgrcmykw'

plt.figure()
for f in range(1, numFeatures+1):
    plt.plot(x, oobErrors[f-1], color=colors[f-1], lw=2, label=f)
plt.xlabel('Number of Trees')
plt.ylabel('Out of Bag Error')
plt.title('Out of bag error vs. Number of trees')
plt.legend(loc="upper left")

plt.figure()
for f in range(1, numFeatures+1):
    plt.plot(x, testRMSEs[f-1], color=colors[f-1], lw=2, label=f)
plt.xlabel('Number of Trees')
plt.ylabel('Test RMSE')
plt.title('Test RMSE vs. Number of trees')
plt.legend(loc="upper right")
print('-------------------------------------')

# iii
bestNumTrees = 27
bestNumFeatures = 3
numTrees = 20
gap = 10
numImpurities = 6
oobErrors = np.zeros((numImpurities, numTrees))
testRMSEs = np.zeros((numImpurities, numTrees))
for t in range(1, numTrees+1):
    print('Tree ' + str(t) + ':')
    for i in range(1, numImpurities + 1):
        print(i)
        rf = RandomForestRegressor(n_estimators=t*gap, max_features=bestNumFeatures, max_depth=4, bootstrap=True, oob_score=True, min_impurity_decrease=1.0*(i-1)/25)
        cv = cross_validate(rf, X, y, cv=10, return_train_score=True, scoring=scoring)
        testRMSEs[i-1][t-1] = math.sqrt(cv['test_mse'].mean())
        rf.fit(X, y)
        oobErrors[i-1][t-1] = rf.oob_score_
moob = np.argmax(oobErrors)
roob, coob = moob//numTrees+1, moob%numTrees+1
mrmse = np.argmin(testRMSEs)
rrmse, crmse = mrmse//numTrees+1, mrmse%numTrees+1
print('Maximum OOB errors: ' + str(oobErrors[roob-1][coob-1]) + ' tree ' + str(coob) + ' min impurity decrease ' + str(roob))
print('Minimum test RMSE: ' + str(testRMSEs[rrmse-1][crmse-1]) + ' tree ' + str(crmse) + ' min impurity decrease ' + str(rrmse))

x = range(10, numTrees*gap+1, 10)
colors = 'bgrcmykw'

plt.figure()
for i in range(1, numImpurities+1):
    plt.plot(x, oobErrors[i-1], color=colors[i-1], lw=1, label=i)
plt.xlabel('Number of Trees')
plt.ylabel('Out of Bag Error')
plt.title('Out of bag error vs. Number of trees')
plt.legend(loc="upper left")

plt.figure()
for i in range(1, numImpurities+1):
    plt.plot(x, testRMSEs[i-1], color=colors[i-1], lw=1, label=i)
plt.xlabel('Number of Trees')
plt.ylabel('Test RMSE')
plt.title('Test RMSE vs. Number of trees')
plt.legend(loc="upper right")
print('-------------------------------------')

# v
rf = RandomForestRegressor(n_estimators=bestNumTrees, max_features=3, max_depth=4, bootstrap=True, oob_score=True)
rf.fit(X, y)
tree = rf.estimators_[0]
y_pred = tree.predict(X)
export_graphviz(tree, out_file='rf.dot')

# Plot the results
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 1].tolist(), X[:, 4].tolist(), y, s=8, c="darkorange", marker='o', label='data')
ax.scatter(X[:, 1].tolist(), X[:, 4].tolist(), y_pred, s=8, c="cornflowerblue", marker='o', label='prediction')

ax.set_xlabel('Day of Week')
ax.set_ylabel('File Name')
ax.set_zlabel('Size of Backup (GB)')
plt.legend()
plt.show()