
# coding: utf-8

# In[5]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from math import log
from sklearn.metrics import confusion_matrix
import itertools

comp_tech_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
rec_act_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
science_subclass = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
miscellaneous_subclass = ['misc.forsale']
politics_subclass = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']
religion_subclass = ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']

dataset = fetch_20newsgroups(subset='all', categories=comp_tech_subclasses + rec_act_subclasses + science_subclass + miscellaneous_subclass + politics_subclass + religion_subclass, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
labels = dataset.target

# seperate all data into 6 classes
labels = [0]*len(dataset.data)
for i in range(len(dataset.data)):
    curtarget = dataset.target[i]
    if curtarget > 4 and curtarget <= 8:
        labels[i] = 1
    elif curtarget > 8 and curtarget <= 12:
        labels[i] = 2
    elif curtarget > 12 and curtarget <= 13:
        labels[i] = 3
    elif curtarget > 13 and curtarget <= 16:
        labels[i] = 4
    else:
        labels[i] = 5

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

vectorizer = TfidfVectorizer(min_df=3, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)

# reduce dimension
nums = [1, 2, 3, 5, 10, 20, 50, 100, 300]

# approach1: SVD + Normalizing features
print 'SVD + Normalizing'
svd = TruncatedSVD(n_components = 300, n_iter = 13,random_state = 42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
temp_res = lsa.fit_transform(X)
svd_res = []
for index in nums:
    svd_res.append(temp_res[:, 0:index])

# approach2: NMF + non-linear transformation
print 'NMF + non-linear transformation'
nmf_res = []
for terms in nums:
    nmf = NMF(n_components=terms)
    X_reduced = nmf.fit_transform(X)
    for j in range(X_reduced.shape[0]):
        for k in range(X_reduced.shape[1]):
            if X_reduced[j][k] == 0:
                X_reduced[j][k] = -3.08
            else:
                X_reduced[j][k] = log(X_reduced[j][k], 10)
    nmf_res.append(X_reduced)

measures = ['Homogeneity Score', 'Completeness Score', 'V-measure', 'Adjusted Rand Score', 'Adjusted Mutual Information Score']
method = ['Truncated SVD', 'NMF']
# plot 5 measures
def plot_histogram(reduce_dimension_method, measure, ydata):
    x_labels = ['1', '2', '3', '5', '10', '20', '50', '100', '300']
    fig, ax = plt.subplots()
    color = ['b', 'g', 'r', 'c', 'm', 'darkorange', 'k', 'pink', 'deepskyblue']
    ax.set_xticks([i+0.25 for i in range(1,10)])
    ax.set_xticklabels(x_labels, fontsize = 12)
    
    rects = plt.bar([i for i in range(1,10)], ydata, 0.5, align='edge', alpha = 0.8, color = color)
    plt.xlabel('Number of Principal Components r', fontsize = 14)
    plt.ylabel('Measure Score', fontsize = 14)
    plt.title(measure+' ('+reduce_dimension_method+')', fontsize = 18)
    plt.axis([0.5,10,0,1])
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height, '%.3f' % float(height), ha='center', va='bottom')
    
    plt.show()

# trasform confusion matrix to diagonal as much as possible (only for 2 * 2 sized confusion)
def to_diagonal(confusion):
    maxColIndices = []
    copy = []
    for row in range(len(confusion)):
        curRow = confusion[row]
        index = 0
        value = curRow[0]
        ro = []
        for col in range(len(curRow)):
            ro.append(curRow[col])
            if curRow[col] > value:
                index = col
                value = curRow[col]
        maxColIndices.append(index)
        copy.append(ro)
    res = []
    for i in range(len(confusion)):
        res.insert(maxColIndices[i], copy[i])
    res = np.array(res)
    return res

# contingency table (confusion matrix)
actual_class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
cluster_class_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6']
def plot_contingency_table(cm, title='Contingency table', cmap=plt.cm.YlOrBr):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(actual_class_names))
    plt.xticks(tick_marks, actual_class_names)
    plt.yticks(tick_marks, cluster_class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Cluster Class', fontsize=12)
    plt.xlabel('Actual Class', fontsize=12)    

# plot clustering results
def plot_clusters(actual_labels, clustered_labels, X_2d, centers, reducer):
    plt.figure(figsize=(12,9))
    color = ["deeppink", "deepskyblue", "orange", "pink", "mediumturquoise", "tan"]
    mark = ["o", "x", "s", "*", "D", "p"]
    for i in range(len(labels)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], s=12, marker=mark[actual_labels[i]], color=color[clustered_labels[i]], alpha=0.5)
    for i in range(6):
        plt.scatter(centers[i, 0], centers[i, 1], marker='+', s=150, linewidths=20, color='k', alpha=0.8)
    plt.title('Clustering Visualization Results (' + reducer + ')', fontsize=23)
    plt.show()

# K-Means clustering
def k_means(data, nums, improve_method, labels):
    homogeneity = []
    completeness = []
    v_measure = []
    rand_score = []
    mutual_info_score = []
    plots = []
    plots.append(homogeneity)
    plots.append(completeness)
    plots.append(v_measure)
    plots.append(rand_score)
    plots.append(mutual_info_score)
    contin = []

    for index in range(len(nums)):
        km = KMeans(n_clusters=6)
        km.fit(data[index])
        confusion = metrics.confusion_matrix(labels, km.labels_)
        contingency = to_diagonal(confusion)
        contin.append(contingency)
        homogeneity.append(metrics.homogeneity_score(labels, km.labels_))
        completeness.append(metrics.completeness_score(labels, km.labels_))
        v_measure.append(metrics.v_measure_score(labels, km.labels_))
        rand_score.append(metrics.adjusted_rand_score(labels, km.labels_))
        mutual_info_score.append(metrics.adjusted_mutual_info_score(labels, km.labels_))
        # plot clustering visualization results
        if index == 1:
            clustered_labels = km.labels_
            centers = km.cluster_centers_
            # plot clutering results
            plot_clusters(labels, clustered_labels, data[index], centers, improve_method)
    return plots, contin

improve = ['SVD + Normalizing', 'NMF + non-linear transformation']
# SVD
res_svd = k_means(svd_res, nums, improve[0], labels)
# plot 5 measures scores
for i in range(5):
    plot_histogram(method[0], measures[i], res_svd[0][i])
# Plot non-normalized contingency table
for i in range(9):
    plt.figure()
    title = 'Contingency table (Truncated SVD, r = ' + str(nums[i]) + ')'
    plot_contingency_table(res_svd[1][i], title=title)
    plt.show()

# NMF
res_nmf = k_means(nmf_res, nums, improve[1], labels)
# plot 5 measures scores
for i in range(5):
    plot_histogram(method[1], measures[i], res_nmf[0][i])
# Plot non-normalized contingency table
for i in range(9):
    plt.figure()
    title = 'Contingency table (NMF, r = ' + str(nums[i]) + ')'
    plot_contingency_table(res_nmf[1][i], title=title)
    plt.show()

    

