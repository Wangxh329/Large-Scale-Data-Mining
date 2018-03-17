
# coding: utf-8

# # Step1: preprocess data

# In[19]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz


# --------------------- preprocessing ----------------------- #
# define paths
files = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt']

# calculate statistics of each hashtag
def cal_statistics(file):
    date = []
    time = []
    tweet_count = []
    followers_count = []
    retweet_count = []
    url_count = []
    author_time = {} # name+nick : date : set(time)
    authors_count = [] 
    mentions_count = []
    rank_score = []
    hashtag_count = []
    # extract data
    with open(file, 'r') as cur_file:
        for line in cur_file:
            data = json.loads(line)
            # date and time
            timestamp = data['citation_date']
            pst_tz = pytz.timezone('US/Pacific')
            timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
            date_split = timestamp[0:10].split('-')
            cur_date = int(date_split[0]+date_split[1]+date_split[2])
            date.append(cur_date)
            cur_time = int(timestamp[11:13])
            time.append(cur_time)
            
            tweet_count.append(1)
            followers_count.append(data['author']['followers'])
            retweet_count.append(data['metrics']['citations']['total'])
            url_count.append(len(data['tweet']['entities']['urls']))
            
            # unique authors
            author_name = data['author']['name']+'+'+data['author']['nick']
            if author_name in author_time:
                ori_ = author_time[author_name]
                if cur_date in ori_:
                    ori_times = ori_[cur_date] # set
                    if cur_time in ori_times:
                        authors_count.append(0)
                    else:
                        authors_count.append(1)
                        ori_times.add(cur_time)
                else:
                    authors_count.append(1)
                    new_times = set()
                    new_times.add(cur_time)
                    ori_[cur_date] = new_times
            else:
                authors_count.append(1)
                new_times = set()
                new_times.add(cur_time)
                new_dates = {}
                new_dates[cur_date] = new_times
                author_time[author_name] = new_dates
                
            mentions_count.append(len(data['tweet']['entities']['user_mentions']))
            rank_score.append(data['metrics']['ranking_score'])
            hashtag_count.append(data['title'].count('#'))
        df = pd.DataFrame({
            'tweet' : tweet_count,
            'date' : date,
            'time' : time,
            'followers' : followers_count,
            'retweets' : retweet_count,
            'urls' : url_count,
            'authors' : authors_count,
            'mentions' : mentions_count,
            'ranking score' : rank_score,
            'hashtags' : hashtag_count
        }, columns = ['tweet', 'date', 'time', 'followers', 'retweets', 'urls', 'authors', 'mentions', 'ranking score', 'hashtags'])
        df.to_csv('extracted_data/Q1.3_'+file[18:-4]+'.csv', index = False)

# extract data from each hashtag
for file in files:
    cal_statistics(file)


# # Step2: train linear regression model

# In[38]:


import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults


# --------------------- train linear regression model ----------------------- #
# define paths
files = ['extracted_data/Q1.3_#gohawks.csv', 'extracted_data/Q1.3_#gopatriots.csv', 'extracted_data/Q1.3_#nfl.csv', 'extracted_data/Q1.3_#patriots.csv', 'extracted_data/Q1.3_#sb49.csv', 'extracted_data/Q1.3_#superbowl.csv']

# calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# load and process data from each hashtag file
def load_and_process(file):
    # process and groupby data
    data = pd.read_csv(file)
    data.columns = ['tweet', 'date', 'time', 'followers', 'retweets', 'urls', 'authors', 'mentions', 'ranking score', 'hashtags']
    df = data.groupby(['date', 'time']).agg({'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'urls' : np.sum, 'authors' : np.sum, 'mentions' : np.sum, 'ranking score' : np.sum, 'hashtags' : np.sum})
    
    # fill up non-exists hours with all zero data
    app_rows = []
    for i in range(1,len(df.index)):  
        pre_date = df.index[i-1][0]
        pre_hour = int(df.index[i-1][1])
        cur_date = df.index[i][0]
        cur_hour = int(df.index[i][1])
        if (cur_hour < pre_hour):
            cur_hour = cur_hour + 24
        hour_diff = cur_hour - pre_hour
        while (hour_diff > 1):
            pre_hour = pre_hour + 1
            if (pre_hour > 23):
                pre_date = cur_date
                app_rows.append({'tweet':0,'date':pre_date,'time':pre_hour-24,'followers':0,'retweets':0,'urls':0,'authors':0,'mentions':0,'ranking score':0,'hashtags':0})
            else:
                app_rows.append({'tweet':0,'date':pre_date,'time':pre_hour,'followers':0,'retweets':0,'urls':0,'authors':0,'mentions':0,'ranking score':0,'hashtags':0})
            hour_diff = cur_hour - pre_hour
    for row in app_rows:
        data = data.append(row, ignore_index=True)
    
    df = data.groupby(['date', 'time']).agg({'time' : np.max, 'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'urls' : np.sum, 'authors' : np.sum, 'mentions' : np.sum, 'ranking score' : np.sum, 'hashtags' : np.sum})
    df.to_csv('extracted_data/Q1.3_hourly_'+file[20:-4]+'.csv', index=False)
    display(df)
    return df

# train and fit linear regression model
def regression_analysis(file, df):
    input_arr = []
    for index in df.index:
        input_arr.append(df.loc[index, 'tweet':'hashtags'].values)
    input_arr.pop()
    input_arr = sm.add_constant(input_arr)
    output_arr = df.loc[df.index[1]:, 'tweet'].values
    
    model = sm.OLS(output_arr, input_arr)
    results = model.fit()
    output_predicted = results.predict(input_arr)
    
    print ('============================================ '+file[20:-4]+' ==============================================')
    
#     # test
#     df2 = pd.DataFrame({
#         'true value':output_arr,
#         'predict value':output_predicted
#     })
#     display(df2)
#     # test
    
    # RMSE
    rmse_ = rmse(output_predicted, output_arr)
    print ('RMSE of the linear regression model is: '+str(rmse_))
    
    # plot fitted values vs true values
    plt.figure(figsize=(15,9))
    plt.scatter(output_arr, output_predicted, color='deeppink', edgecolors='k')
    plt.plot([output_arr.min(), output_arr.max()], [output_arr.min(), output_arr.max()], 'k--', lw=4)
    plt.ylabel('Fitted Number of Tweets in Next Hour', fontsize = 18)
    plt.xlabel('True Number of Tweets in Next Hour', fontsize = 18)
    plt.title('Fitted Values vs True Values ('+file[20:-4]+')', fontsize = 23)
    plt.show()
    
    # regression analysis
    print (results.summary())
    return output_predicted

# 'tweet','retweets','followers','urls','authors','mentions','ranking score','hashtags'
#   1         2           3        4        5          6           7             8
top3_features = {'#gohawks':['followers','mentions','authors'], 
                 '#gopatriots':['retweets','ranking score','authors'], 
                 '#nfl':['hashtags','authors','mentions'], 
                 '#patriots':['retweets','hashtags','mentions'], 
                 '#sb49':['followers','hashtags','mentions'], 
                 '#superbowl':['retweets','hashtags','ranking score']}
# plot top 3 features results
def plot_top3_features(file, df, output_predicted):
    features = top3_features[file]
    opt = []
    for i in range(3):
        cur_feature = []
        for index in df.index:
            cur_feature.append(df.loc[index, features[i]:features[i]].values)
        cur_feature.pop()
        opt.append(cur_feature)
    
    fig = plt.figure(figsize=(16,5))
    fig.suptitle(file, y=1.07, fontsize=23)
    for i in range(3):
        ax = plt.subplot(131+i)
        ax.set_title('Predictant vs '+features[i], fontsize=18)
        ax.scatter(opt[i], output_predicted, color='deeppink', edgecolors='k')
        ax.set_xlabel(features[i], fontsize=14)
        ax.set_ylabel('predicted number of tweets', fontsize=14)
    plt.tight_layout()
    plt.show()
    print ('==========================================================================================================')
    print ()

# linear regression model on specified hashtag
def regression_model(file):
    df = load_and_process(file)
    output_predicted = regression_analysis(file, df)
    plot_top3_features(file[20:-4], df, output_predicted)

# linear regression model on each hashtag
for file in files:
    regression_model(file)

