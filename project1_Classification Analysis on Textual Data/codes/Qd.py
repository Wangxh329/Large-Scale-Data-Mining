
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import re

# define two classes
computer_technology_class = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
recreational_activity_class = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
training_data = fetch_20newsgroups(subset='train', categories=computer_technology_class+recreational_activity_class, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
stemmer = SnowballStemmer("english")

# process data
def trim_and_stem(data_list):
    for i in range(len(data_list)):
        temp = re.findall('[a-zA-Z]+', data_list[i])
        ans = []
        for j in range(len(temp)):
            if not temp[j].isdigit():
                ans.append(stemmer.stem(temp[j])) # stem() turned words into lowercase            
        data_list[i] = " ".join(ans)
trim_and_stem(training_data.data)

# generate TFxIDF
count_vect = CountVectorizer(min_df=2, stop_words ='english')
X_counts = count_vect.fit_transform(training_data.data)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# using LSI
svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
svd_res = svd.fit_transform(X_tfidf)
print 'LSI method:'
print svd_res.shape
print ''

# using NMF
nmf = NMF(n_components=50, init='random', random_state=0)
nmf_res = nmf.fit_transform(X_tfidf)
print 'NMF method:'
print nmf_res.shape
print ''

