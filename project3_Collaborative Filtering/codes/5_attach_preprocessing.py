
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict

# Load data
dataset = np.genfromtxt('ratings.csv', delimiter=',', skip_header=1, usecols=(0,1,2))
row, col, data = dataset[:, 0], dataset[:, 1], dataset[:, 2]
rows, cols = max(row), max(col)
print('Shape of dataset is {} * {}'.format(rows, cols))

# Trim data
collection = defaultdict(list)
for i in range(len(col)):
    collection[col[i]].append((row[i], data[i]))
pop, unpop, var, binary = [], [], [], [[], [], [], []]
thresholds = [2.5, 3, 3.5, 4]
for c, v in collection.items():
    variance = np.var(list(map(lambda x:x[1], v)))
    for r, d in v:
        for i, threshold in enumerate(thresholds):
            binary[i].append([r, c, int(d >= threshold)])
        if len(v) > 2:
            pop.append([r, c, d])
        else:
            unpop.append([r, c, d])
        if len(v) >= 5 and variance >= 2:
            var.append([r, c, d])

# Save data
np.savetxt('popular.csv', pop, fmt=['%d', '%d', '%.1f'], delimiter=',', header='userId,movieId,rating')
np.savetxt('unpopular.csv', unpop, fmt=['%d', '%d', '%.1f'], delimiter=',', header='userId,movieId,rating')
np.savetxt('variance.csv', var, fmt=['%d', '%d', '%.1f'], delimiter=',', header='userId,movieId,rating')
for i, threshold in enumerate(thresholds):
    np.savetxt('bin'+str(threshold)+'.csv', binary[i], fmt=['%d', '%d', '%d'], delimiter=',', header='userId,movieId,rating')
print('Datasets are written successfully!')

