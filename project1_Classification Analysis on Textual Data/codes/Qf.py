
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
import re
from sklearn.metrics import confusion_matrix
import itertools


# fetch original data
computer_technology_subclasses = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
recreational_activity_subclasses = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
comp_tech_train = fetch_20newsgroups(subset='train', categories=computer_technology_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_train = fetch_20newsgroups(subset='train', categories=recreational_activity_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
comp_tech_test = fetch_20newsgroups(subset='test', categories=computer_technology_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
rec_act_test = fetch_20newsgroups(subset='test', categories=recreational_activity_subclasses, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
all_data = comp_tech_train.data+comp_tech_test.data+rec_act_train.data+rec_act_test.data


# map comp_tech class to True and rec_act class to False
classification_train = [True] * len(comp_tech_train.data) + [False] * len(rec_act_train.data)
classification_test = [True] * len(comp_tech_test.data) + [False] * len(rec_act_test.data)


# trim data
stemmer = SnowballStemmer("english")
def trim_and_stem(data_list):
    for i in range(len(data_list)):
        temp = re.findall('[a-zA-Z]+', data_list[i])
        ans = []
        for j in range(len(temp)):
            if not temp[j].isdigit():
                ans.append(stemmer.stem(temp[j])) # stem() turned words into lowercase            
        data_list[i] = " ".join(ans)
trim_and_stem(all_data)


# generate count matrices with min_df = 2 & 5
X_counts = []
for i in [2, 5]:
    count_vect = CountVectorizer(min_df=i, stop_words ='english')
    X_counts.append(count_vect.fit_transform(all_data))


# generate tfidf matrices 
tfidf_transformer = TfidfTransformer()
X_tfidf = []
for i in [0, 1]:
    X_tfidf.append(tfidf_transformer.fit_transform(X_counts[i]))


# extract feature using LSI & NMF
lsi_res = []
svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
for i in [0, 1]:
    lsi_res.append(svd.fit_transform(X_tfidf[i]))
nmf = NMF(n_components=50, init='random', random_state=0)
nmf_res = nmf.fit_transform(X_tfidf[0])


# split train and test dataset
split_point_1 = len(comp_tech_train.data)
split_point_2 = split_point_1 + len(comp_tech_test.data)
split_point_3 = split_point_2 + len(rec_act_train.data)
lsi_test = []
lsi_train = []
for i in [0, 1]:
    lsi_test.append(np.concatenate((lsi_res[i][split_point_1 : split_point_2], lsi_res[i][split_point_3:])))
    lsi_train.append(np.concatenate((lsi_res[i][0:split_point_1], lsi_res[i][split_point_2:split_point_3])))
nmf_test = np.concatenate((nmf_res[split_point_1 : split_point_2], nmf_res[split_point_3:]))
nmf_train = np.concatenate((nmf_res[0:split_point_1], nmf_res[split_point_2:split_point_3]))
lsi_test += [nmf_test]
lsi_train += [nmf_train]
test = lsi_test
train = lsi_train


# find optimal gamma
method = ['LSI', 'LSI', 'NMF']
min_df = [2, 5, 2]

parameters = {'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

## confusion matrix
class_names = ['Computer\n Technology', 'Recreational\n Activity']
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.YlOrBr):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted Class', fontsize=12)
    plt.xlabel('Actual Class', fontsize=12)

for i in range(3):
    classifier = svm.SVC(probability=True)
    search = GridSearchCV(classifier, parameters, cv=5)
    search.fit(train[i], classification_train)
    news_true, news_predicted = classification_test, search.predict(test[i])
    
    print 'min_df = ' + str(min_df[i])
    print 'dimension reduction method: ' + str(method[i])
    print "best parameter:"
    print search.best_params_
    print "accuracy:"
    print metrics.accuracy_score(news_true, news_predicted)
    print "precision:"
    print metrics.precision_score(news_true, news_predicted)
    print "recall:"
    print metrics.recall_score(news_true, news_predicted)
    print "confusion matrix:"
    conf_mat = metrics.confusion_matrix(news_true, news_predicted)
    print conf_mat
    # Plot non-normalized confusion matrix
    plt.figure()
    title = 'Confusion Matrix ('+method[i]+', min_df='+str(min_df[i])+', gamma='+str(search.best_params_)+')'
    plot_confusion_matrix(conf_mat, classes=class_names, title=title)
    plt.show()

