
# coding: utf-8

# In[4]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np


# fetch original data
comp_tech_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']             
rec_act_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset='all', categories=comp_tech_subclasses+rec_act_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

# seperate all data into two classes
labels = [1]*len(dataset.data)
for i in range(len(dataset.data)):
    if dataset.target[i] > 3:
        labels[i] = 0

# trim data
def trim_data(data_list):
    for i in range(len(data_list)):
        temp = re.findall('[a-zA-Z]+', data_list[i])
        ans = []
        for j in range(len(temp)):
            if not temp[j].isdigit():
                ans.append(temp[j])           
        data_list[i] = " ".join(ans)
        
trim_data(dataset.data)

# generate TF-IDF matrix
vectorizer = TfidfVectorizer(min_df=3, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)

# calculate singular values
svd = TruncatedSVD(n_components = X.shape[1] - 1, n_iter = 10, random_state = 42)
lsi_res = svd.fit_transform(X)

singular_values = svd.singular_values_
sin_val_square = [x*x for x in singular_values]
original_variance = np.sum(sin_val_square)
prefix = []
percent = []
prefix.append(sin_val_square[0])
percent.append(prefix[0] / original_variance)
for terms in range(1, 1000, 1):
    prefix.append(prefix[terms - 1] + sin_val_square[terms])
    percent.append(prefix[terms] / original_variance)

# calculate percent of variance v.s. r
x_axis = [x for x in range(1, 1001, 1)]
plt.figure(figsize=(12,9))
plt.xlabel('Number of Principal Components r', fontsize = 18)
plt.ylabel('Percent of Variance Retained', fontsize = 18)
plt.title('Percent of Variance Retained (r = 1 to 1000)', fontsize = 23)
plt.xlim([0.0, 1010])
plt.ylim([0.0, 0.6])
plt.plot(x_axis, percent)
plt.show()

