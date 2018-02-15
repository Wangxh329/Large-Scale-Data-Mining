
# coding: utf-8

# In[28]:


from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re

stemmer = SnowballStemmer("english")

# merge documents in one class into a document
def join_data(category, subset):
    data_list = fetch_20newsgroups(subset=subset, categories=[category], shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')).data
    return [" ".join(data_list)]

def trim_and_stem(data_list):
    for i in range(len(data_list)):
        temp = re.findall('[a-zA-Z]+', data_list[i])
        ans = []
        for j in range(len(temp)):
            if not temp[j].isdigit():
                ans.append(stemmer.stem(temp[j])) # stem() turned words into lowercase            
        data_list[i] = " ".join(ans)

def ten_most(subset, categories, indices):
    print 'Subset: ' + subset
    # merge classes
    classes = []
    for category in categories:
        data_list = fetch_20newsgroups(subset=subset, categories=[category], shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')).data
        trim_and_stem(data_list)
        classes += join_data(category, subset)
    
    # count words and generate tfidf
    count_vect = CountVectorizer(min_df=2, stop_words ='english')
    X_counts = count_vect.fit_transform(classes)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts).toarray()
    
    # sort words according to their importance
    global_res = []
    global_sig = []
    for index in indices:
        print '10 most significant terms in \"' + categories[index]+ '\" (subset: ' + subset + '):'
        sorted_index = np.argsort(X_tfidf[index])
        res = []
        sig = []
        i = 1
        while len(res) < 10:
            cur = count_vect.get_feature_names()[sorted_index[-i]]
            utf8string = cur.encode("utf-8")
            if (not utf8string.isdigit()):
                res.append(utf8string)
                sig.append(X_tfidf[index][sorted_index[-i]])
            i += 1
        print res
        print sig
        print ''
        global_res.append(res)
        global_sig.append(sig)
    return global_res, global_sig

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
            'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
            'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 
            'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 
            'talk.politics.misc', 'talk.religion.misc']
indices = [3, 4, 6, 15]

for subset in ['test', 'all', 'train']:
    pair = ten_most(subset, categories, indices)
    res = pair[0]
    sig = pair[1]
    # plot histogram
    fig = plt.figure(figsize=(16,12))
    figure_title = '10 most significant terms (subset: ' + subset + ')'
    fig.suptitle('10 most significant terms (subset: ' + subset + ')', y=1.03, fontsize=23)
    for i in range(4):
        ax = plt.subplot(221+i)
        ax.set_title('\"' + categories[indices[i]]+ '\"', fontsize=18)
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'darkorange', 'deepskyblue']
        ax.set_xticks([j+0.25 for j in range(1, 11)])
        ax.set_xticklabels(res[i], rotation=45, ha='right', fontsize=12)
        rects = plt.bar([j for j in range(1, 11)], sig[i], 0.5, align='edge', alpha = 0.8, color = color)
        ax.set_xlabel('10 most significant terms', fontsize=14)
        ax.set_ylabel('Significance', fontsize=14)
        ax.axis([0.5,11,0,0.5])
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1*height, '%.2f' % float(height), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

