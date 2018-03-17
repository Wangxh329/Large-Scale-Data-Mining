import numpy as np
import json
import pandas as pd
import math
import datetime, time
import pytz
from pytz import timezone
import nltk
import calendar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def clean_text_words(text):
    #clean words
    # split into words
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    #remove the word after hashtag
    idx= []
    tokens1 = []
    for i in range(len(tokens)):
        if i == 0:
            idx.append(0)
        elif (tokens[i-1] != '#'):
                idx.append(i)
        pass
    pass
    for i in idx:
        tokens1.append(tokens[i])
    pass

    # remove the symbol
    words = [w for w in tokens1 if w.isalpha()]
    #remove customized word
    words1 = [w for w in words if not w in ['http']]

    #remove stopwords
    stop_words = stopwords.words('english')
    words = [w for w in words1 if not w in stop_words]

    #combine stem words together
    porter = PorterStemmer()
    stemmed = [porter.stem(w) for w in words]
    return stemmed


f = open('tweets_#gopatriots.txt', 'r')#open('tweets_#gohawks.txt', 'r')
timestamp = []
tweet_count = []
followers_count = []
retweet_count = []
content = []
with f as cur_file:
    for line in cur_file:
    #json_object = json.loads(line)
        data = json.loads(line)
        timestamp.append(data['citation_date'])
        tweet_count.append(1)
        followers_count.append(data['author']['followers'])
        retweet_count.append(data['metrics']['citations']['total'])
        content.append(data['title'])
    pass
pass

df_new = pd.DataFrame({
    'tweet': tweet_count,
    'timestamp': timestamp,
    'followers': followers_count,
    'retweeted times': retweet_count,
    'content': content},
    columns=['tweet', 'timestamp', 'followers', 'retweeted times', 'content'])


#clean the word and make sentence to do pos/neg analysis
combined_word=[]
for i in range(df_new.shape[0]):#
    stemmed = clean_text_words(df_new['content'][i])#
    # join words to a sentence
    combined_word.append(' '.join(w for w in stemmed))
    print(i)
pass

sid = SentimentIntensityAnalyzer()
neg_portion=[]
pos_portion=[]
neu_portion=[]
for i in range(df_new.shape[0]):#
    ss = sid.polarity_scores(combined_word[i])
    neg_portion.append(ss['neg'])
    pos_portion.append(ss['pos'])
    neu_portion.append(ss['neu'])
    print(i)
pass

ss = sid.polarity_scores(data['title'])
#type(ss['neg']) ->float

#create a dataframe save the pos, negative and timestamp to further analyze
df_new1 = pd.DataFrame({
    'pos_pot': pos_portion,
    'neg_pot': neg_portion,
    'neu_pot': neu_portion,
    'timestamp': timestamp,
    'combined_word': combined_word
    },
    columns=['pos_pot', 'neg_pot', 'neu_pot', 'timestamp', 'combined_word'])

#define the 'sentence' as pos ot negative
df_new = df_new1

pos_ov = []
neg_ov = []
for i in range(df_new.shape[0]):
    if df_new['pos_pot'][i] > df_new['neg_pot'][i]:
        pos_ov.append(1)
        neg_ov.append(0)
    elif df_new['pos_pot'][i] < df_new['neg_pot'][i]:
        pos_ov.append(0)
        neg_ov.append(1)
    else:
        pos_ov.append(0)
        neg_ov.append(0)
    pass
pass
df_new['pos_ov'] = pos_ov
df_new['neg_ov'] = neg_ov

#from 2015-2-1 13:00 8 hrs
pos_arr=[]
neg_arr=[]
pos_cnt=[]
neg_cnt=[]
cnt=[]
for i in range(0, 10* 48+1, 10):

    dt_low = datetime.datetime(2015, 2, 1, 13 + i//60, i%60)
    dt_high = datetime.datetime(2015, 2, 1, 13 + (i+10)//60 , (i+10)%60)
    ts_threshold_low = calendar.timegm(dt_low.utctimetuple()) + 8 * 3600
    ts_threshold_high = calendar.timegm(dt_high.utctimetuple()) + 8 * 3600
    idc = np.where((df_new['timestamp']>ts_threshold_low)&(df_new['timestamp'] < ts_threshold_high))[0]#6:30pm 3.37 long
    pos_arr.append(np.mean(df_new['pos_pot'][idc]))
    neg_arr.append(np.mean(df_new['neg_pot'][idc]))

    cnt.append(idc.shape[0])
    pos_cnt.append(np.sum(df_new['pos_ov'][idc])/ idc.shape[0])
    neg_cnt.append(np.sum(df_new['neg_ov'][idc])/ idc.shape[0])
pass

#try pos - neg
cnt_hawks = np.copy(cnt)
pos_cnt_hawks = np.copy(pos_cnt)
neg_cnt_hawks = np.copy(neg_cnt)
pos_cnt_hawks_nor = np.copy((pos_cnt - np.mean(df_new['pos_ov'])) / np.std(df_new['pos_ov']))
neg_cnt_hawks_nor = np.copy((neg_cnt - np.mean(df_new['neg_ov'])) / np.std(df_new['neg_ov']))

pos_arr_hawks = np.copy(pos_arr)
neg_arr_hawks = np.copy(neg_arr)
pos_arr_hawks_nor = np.copy((pos_arr - np.mean(df_new['pos_pot']))/ np.std(df_new['pos_pot']))
neg_arr_hawks_nor = np.copy((neg_arr - np.mean(df_new['neg_pot']))/np.std(df_new['neg_pot']))

#plot the number of counts in each time period
plt.plot(cnt)
#plt.plot([0, 50],
#         [500, 500], 'k--', lw=2)
plt.title('Comment counts vs time in 2/1/2015 #gopatriots')
plt.ylabel('count')
plt.xlabel('timestamp: 10 min an index from 13:00')
plt.show()

#plot pos and negative (as words)
#plt.plot(pos_arr)
#plt.plot(neg_arr)
#plt.plot(pos_arr_hawks)
#plt.plot(neg_arr_hawks)
#plt.plot(pos_cnt)
#plt.plot(neg_cnt)
plt.plot(pos_cnt_hawks)
plt.plot(neg_cnt_hawks)
plt.title('Positive and Negative comment in 2/1/2015 #gohawks')#gopatriots
plt.ylabel('probability')
plt.xlabel('timestamp: 10 min an index from 13:00')
plt.legend(['positive', 'negative'])
plt.show()


#plot normalized plot(as words)
#plt.plot((pos_cnt - np.mean(df_new['pos_ov'])) / np.std(df_new['pos_ov'])) #normalized plot
#plt.plot((neg_cnt - np.mean(df_new['neg_ov'])) / np.std(df_new['neg_ov']))
plt.plot(pos_cnt_hawks_nor)
plt.plot(neg_cnt_hawks_nor)
plt.ylabel('number of std ')
plt.xlabel('timestamp: 10 min an index from 13:00')
plt.title('Positive and Negative comment(normalized) in 2/1/2015 #gohawks')
plt.legend(['positive', 'negative'])
plt.show()

#plot (pos-neg) plot
#plot (pos-neg) normalized plot


go_patriots_posneg = ((pos_cnt - np.mean(df_new['pos_ov'])) / np.std(df_new['pos_ov']) -
                        (neg_cnt - np.mean(df_new['neg_ov'])) / np.std(df_new['neg_ov']))
plt.plot(pos_cnt_hawks_nor - neg_cnt_hawks_nor)
plt.plot(go_patriots_posneg)

plt.legend(['#gohawks', '#gopatriots'])
plt.xlabel('timestamp: 10 min an index from 13:00')
plt.title('(Positive - Negative)comment in 2/1/2015')
plt.show()


#Get increase or decrease
temp_a=[]
temp_b=[]
for i in range(48):
    temp_a.append(np.sign(go_patriots_posneg[i+1] -go_patriots_posneg[i]))
    temp_b.append(np.sign((pos_cnt_hawks_nor - neg_cnt_hawks_nor)[i+1]-(pos_cnt_hawks_nor - neg_cnt_hawks_nor)[i]))
pass
temp_c = np.array(temp_a) * np.array(temp_b)
np.where(temp_c==-1)[0].shape


# hawk loss
# number of twittering after the game, analysis text
# before game, during game, after game, two teams
# Seahawks is schedule for 6:30 p.m. EST -3
# when game go tense, the sentiment different
# Football games last for a total of 60 minutes in professional and college play
# and are divided into two-halves of 30 minutes and four-quarters of 15 minutes.

#game over, hawks decrease 41-48 3 times, patriots decrease 1 and small
#After the game, analyze probability, patriots go increase a lot, hawks not. patriots win,
#  last 2 timstamp, cnt so small no discuss

#save and load
#df_new.to_pickle('/Users/Ray/Desktop/ECE219/dfnew_hawk_w_stem')
#df_new = pd.read_pickle('/Users/Ray/Desktop/ECE219/dfnew_hawk_w_stem')