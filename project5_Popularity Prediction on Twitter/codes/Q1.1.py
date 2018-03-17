
# coding: utf-8

# In[14]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# define paths
files = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt']

# plot figure
def plot_histogram(timestamp, file):
    tweets_in_hour = [0] * int((max(timestamp)-min(timestamp))/3600+1)
    start = min(timestamp)
    for time in timestamp:
        tweets_in_hour[int((time-start)/3600)] += 1
    plt.figure(figsize=(15,9))
    plt.bar([i for i in range(0,len(tweets_in_hour))], tweets_in_hour, 1, align='edge', color = 'deepskyblue')
    plt.xlabel('Time/hour', fontsize = 18)
    plt.ylabel('Number of Tweets', fontsize = 18)
    plt.title('Number of Tweets in Hour (' + file[18:-4] + ')', fontsize = 23)
    plt.show()

# calculate statistics of each hashtag
def cal_statistics(file):
    timestamp = []
    tweet_count = []
    followers_count = []
    retweet_count = []
    # extract data
    with open(file, 'r') as cur_file:
        for line in cur_file:
            data = json.loads(line)
            timestamp.append(data['citation_date'])
            tweet_count.append(1)
            followers_count.append(data['author']['followers'])
            retweet_count.append(data['metrics']['citations']['total'])
        df = pd.DataFrame({
            'tweet' : tweet_count,
            'timestamp' : timestamp,
            'followers' : followers_count,
            'retweeted times' : retweet_count
        }, columns = ['tweet', 'timestamp', 'followers', 'retweeted times'])
        df.to_csv('extracted_data/Q1.1_'+file[18:-4]+'.csv', index = False)
        # output statistics result
        print ('===================================================================')
        print ('Hashtag: '+file[18:-4])
        print ('  Average number of tweets per hour: '+str(float(len(tweet_count))/((max(timestamp)-min(timestamp))/3600.0)))
        print ('  Average number of followers of users posting the tweets: '+str(sum(followers_count)/float(len(tweet_count))))
        print ('  Average number of retweets: '+str(sum(retweet_count)/float(len(tweet_count))))
        if file in ['tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#superbowl.txt']:
            plot_histogram(timestamp, file)
        print ('===================================================================')

# calculate each hashtag
for file in files:
    cal_statistics(file)

