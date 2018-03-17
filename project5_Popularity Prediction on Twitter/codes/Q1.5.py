
# coding: utf-8

# # Step1: preprocessing test data from raw data

# In[5]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz


# --------------------- preprocessing test data (raw -> csv) ----------------------- #

# define paths
files_raw = ['test_data/sample1_period1.txt', 'test_data/sample2_period2.txt', 'test_data/sample3_period3.txt', 
             'test_data/sample4_period1.txt', 'test_data/sample5_period1.txt', 'test_data/sample6_period2.txt', 
             'test_data/sample7_period3.txt', 'test_data/sample8_period1.txt', 'test_data/sample9_period2.txt',
             'test_data/sample10_period3.txt']

# calculate statistics of each file
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
            timestamp = data['firstpost_date']
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
        df.to_csv('extracted_data/Q1.5_'+file[10:-4]+'.csv', index = False)

# extract data from each file
for file in files_raw:
    cal_statistics(file)
print ('Raw test data has been done!')


# --------------------- preprocessing test data (csv -> hourly grouped csv) ----------------------- #

# define paths
files_hour = ['extracted_data/Q1.5_sample1_period1.csv', 'extracted_data/Q1.5_sample2_period2.csv', 
              'extracted_data/Q1.5_sample3_period3.csv', 'extracted_data/Q1.5_sample4_period1.csv', 
              'extracted_data/Q1.5_sample5_period1.csv', 'extracted_data/Q1.5_sample6_period2.csv', 
              'extracted_data/Q1.5_sample7_period3.csv', 'extracted_data/Q1.5_sample8_period1.csv', 
              'extracted_data/Q1.5_sample9_period2.csv', 'extracted_data/Q1.5_sample10_period3.csv']

# load and process data from each test file
def load_and_process(file):
    # process and groupby data
    data = pd.read_csv(file)
    data.columns = ['tweet', 'date', 'time', 'followers', 'retweets', 'urls', 'authors', 'mentions', 'ranking score', 'hashtags']
    df = data.groupby(['date', 'time']).agg({'time' : np.max, 'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'urls' : np.sum, 'authors' : np.sum, 'mentions' : np.sum, 'ranking score' : np.sum})
    df.to_csv('extracted_data/Q1.5_hourly_'+file[20:-4]+'.csv', index=False)
    display(df)
    return df

# linear regression model on each file
for file in files_hour:
    load_and_process(file)
print ('Each test data file has been grouped hourly!')


# # Step2: fit best model on train data for each period

# In[91]:


import statsmodels.api as sm
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from statsmodels.regression.linear_model import RegressionResults
from IPython.display import display
from sklearn.metrics import mean_absolute_error


# seperate aggregated train data into three period
# 1. Before Feb. 1, 8:00 a.m.
# 2. Between Feb. 1, 8:00 a.m. and 8:00 p.m.
# 3. After Feb. 1, 8:00 p.m.
def seperate(df):
    periods = []
    periods.append(df.query('date < 20150201 or (date == 20150201 and time < 8)'))
    periods.append(df.query('date == 20150201 and time >= 8 and time <= 20'))
    periods.append(df.query('date > 20150201 or (date == 20150201 and time > 20)'))
    
    return periods

# using the best model in 1.4.2 (all are LR) to train each period (35 features & 28 features for period 1)
def regression_model_for_periods(periods):
    period_model = {}
    # 1 train model for each period (35 features)
    for i in range(3): # 3 periods
        period = periods[i]
        print (len(period.index))
        input_arr = []
        index_start = 0
        for j in range(index_start, index_start+len(period.index)-5): # n-5 points
            cur_input = []
            for k in range(5): # each point has 35 features
                for p in range(2,9): # append each column
                    cur_input.append(period.iloc[j+k, p])
            input_arr.append(cur_input)
        index_start = index_start + len(period.index)

        output_arr = period.loc[period.index[5]:, 'tweet'].values
        
        results = sm.OLS(output_arr, input_arr).fit()
#         if (i == 0):
#             results = svm.SVC(gamma=6)
#             results.fit(input_arr, output_arr)
#         else:
#             results = sm.OLS(output_arr, input_arr).fit()
        period_model[str(i+1)] = results
    
    # 2 train model for period 1 with 28 features
    period1 = periods[0]
    input_arr_ = []
    for j in range(0,len(period1.index)-4): # n-4 points
        cur_input = []
        for k in range(4): # each point has 28 features
            for p in range(2,9): # append each column
                cur_input.append(period1.iloc[j+k, p])
        input_arr_.append(cur_input)
        
    output_arr_ = period1.loc[period1.index[4]:, 'tweet'].values

    results_ = sm.OLS(output_arr_, input_arr_).fit()
#     results_ = svm.SVC(gamma=6)
#     results_.fit(input_arr_, output_arr_)
    period_model['4'] = results_
    
    return period_model

# load data from hourly grouped aggregated train data
df = pd.read_csv('extracted_data/Q1.4_#combine.csv')
df.columns = ['date', 'time', 'tweet', 'retweets', 'followers', 'urls', 'authors', 'mentions', 'ranking score', 'hashtags']
periods = seperate(df)
period_model = regression_model_for_periods(periods)


# # Step3: predict on each test file using corresponding model

# In[93]:


# define paths
files = ['extracted_data/Q1.5_hourly_sample1_period1.csv', 'extracted_data/Q1.5_hourly_sample2_period2.csv', 
         'extracted_data/Q1.5_hourly_sample3_period3.csv', 'extracted_data/Q1.5_hourly_sample4_period1.csv', 
         'extracted_data/Q1.5_hourly_sample5_period1.csv', 'extracted_data/Q1.5_hourly_sample6_period2.csv', 
         'extracted_data/Q1.5_hourly_sample7_period3.csv', 'extracted_data/Q1.5_hourly_sample8_period1.csv', 
         'extracted_data/Q1.5_hourly_sample9_period2.csv', 'extracted_data/Q1.5_hourly_sample10_period3.csv']

# predict on test data
def predict_on_test_data(file, period_model, df):
    period = file[-1]
    input_arr = []
    predicted_output = None
    results = period_model[period]
    cur_input = []
    if file[6] == '8': # 4-hour window
        for i in range(4):
            for p in range(1,8): # append each column
                cur_input.append(df.iloc[i, p])
        results = period_model['4']
    else:
        for i in range(5):
            for p in range(1,8): # append each column
                cur_input.append(df.iloc[i, p])

    input_arr.append(cur_input)
        
    predicted_output = results.predict(input_arr)
    return predicted_output

test_data = []
predict = []
true = []
error = []
# predict on each test file
for file in files:
    test_data.append(file[27:-4])
    
    # load data
    df = pd.read_csv(file)
    df.columns = ['time', 'tweet', 'retweets', 'followers', 'urls', 'authors', 'mentions', 'ranking score']
    
    # predict
    predicted_output = predict_on_test_data(file[27:-4], period_model, df)
    predict.append(predicted_output[0])
    
    # relative error
    true_value = df.loc[df.index[len(df.index)-1], 'tweet':'tweet'].values
    true.append(true_value[0])
    rel_error = abs(predicted_output-true_value)/true_value
    error.append(rel_error[0])
    
res = pd.DataFrame({
    'test file' : test_data,
    'predicted' : predict,
    'true value' : true,
    'relative error' : error
}, columns = ['test file', 'predicted', 'true value', 'relative error'])
display(res)
    

