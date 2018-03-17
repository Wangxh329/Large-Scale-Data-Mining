
# coding: utf-8

# # Step1:  extracting features from raw data

# In[141]:


# generate raw combine dataset
files = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt']

with open('tweet_data/tweets_#combine.txt', 'w') as target:
    for file in files:
        with open(file, 'r') as cur_file:
                for line in cur_file:
                    target.write(line)


# In[142]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz


# --------------------- preprocessing ----------------------- #
# define paths
files = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt', 'tweet_data/tweets_#combine.txt']

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


# # Step2: aggregating data from step1

# In[3]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults


# define paths
files = ['extracted_data/Q1.3_#gohawks.csv', 'extracted_data/Q1.3_#gopatriots.csv', 'extracted_data/Q1.3_#nfl.csv', 'extracted_data/Q1.3_#patriots.csv', 'extracted_data/Q1.3_#sb49.csv', 'extracted_data/Q1.3_#superbowl.csv', 'extracted_data/Q1.3_#combine.csv']

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
    
    df = data.groupby(['date', 'time']).agg({'date' : pd.Series.unique, 'time' : pd.Series.unique, 'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'urls' : np.sum, 'authors' : np.sum, 'mentions' : np.sum, 'ranking score' : np.sum, 'hashtags' : np.sum})
    df.to_csv('extracted_data/Q1.4_'+file[20:-4]+'.csv', index=False)
    return df


# # Step3: seperating data according to date and time
# 
# 1. Before Feb. 1, 8:00 a.m.
# 2. Between Feb. 1, 8:00 a.m. and 8:00 p.m.
# 3. After Feb. 1, 8:00 p.m.

# In[4]:


from IPython.display import display
def seperate(df):
    periods = []
    periods.append(df.query('date < 20150201 or (date == 20150201 and time < 8)'))
    periods.append(df.query('date == 20150201 and time >= 8 and time <= 20'))
    periods.append(df.query('date > 20150201 or (date == 20150201 and time > 20)'))
    return periods


# # Step4: using 3 models to train and predict
# For each hashtag, report the average cross-validation errors for the 3 different models.
# Note that you should do the 90-10% splitting for each model within its specific time
# window.
# <br><br>Your evaluated error should be of the form |Npredicted - Nreal|.
# <br>MAE (mean of 10 absolute errors) for each piece and each model
# <br><br>\- 6 hashtags
# <br>&emsp;&emsp;\- 3 time pieces 
# <br>&emsp;&emsp;&emsp;&emsp;\- **3 models**
# <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\- **10 folds**
# <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\- **average cross-validation error**

# In[7]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model


def regression_analysis(file, periods):
    # input: dataframes of 3 time pieces of a hashtag and the file name
    titles = ['Before', 'Between', 'After']
    res = {}
    res[str(file[20:-4])] = {}
    
    for i in range(len(periods)):
        print('================' + str(file[20:-4]) + ' ' + titles[i] + '================')
        res[str(file[20:-4])][titles[i]] = {}
        period = periods[i]
        input_arr = []
        for index in period.index:
            input_arr.append(period.loc[index, 'tweet':'hashtags'].values)
        input_arr.pop()
        output_arr = period.loc[period.index[1]:, 'tweet'].values
        errors = three_models_ten_folds_errors(input_arr, output_arr)
        for key in errors:
            print(key + ' average error: ' + str(errors[key]))
            res[str(file[20:-4])][titles[i]][key] = errors[key]
    return res


def three_models_ten_folds_errors(input_arr, output_arr):
    ave_error = {}
    ave_error['LR'] = 0
    ave_error['SVM'] = 0
    ave_error['NN'] = 0
    for model in ave_error:
        MAE = []
        kf = KFold(n_splits=10, shuffle=False)
        for train_index, test_index in kf.split(input_arr):
            train_in = [input_arr[i] for i in train_index]
            test_in = [input_arr[i] for i in test_index]
            train_out = [output_arr[i] for i in train_index]
            test_out = [output_arr[i] for i in test_index]
            test_pre = fit_predict(model, train_in, train_out, test_in)
            MAE.append(mean_absolute_error(test_out, test_pre))
        ave_error[model] = np.mean(MAE)
    return ave_error


def fit_predict(model, train_in, train_out, test_in):
    if model == 'LR':
        tr_in = []
        for i in range(len(train_in)):
            tr_in.append(train_in[i][:])
            np.append(tr_in[len(tr_in) - 1], 1)
        te_in = []
        for i in range(len(test_in)):
            te_in.append(test_in[i][:])
            np.append(te_in[len(te_in) - 1], 1)
        reg = sm.OLS(train_out, tr_in)
        results = reg.fit()
        return results.predict(te_in)
    elif model == 'SVM':
        reg = svm.SVC(gamma=6)
        reg.fit(train_in, train_out)
        return reg.predict(test_in)
    elif model == 'NN':
        reg = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu')
        reg.fit(train_in, train_out)
        return reg.predict(test_in)

for file in files:
    df = load_and_process(file)
    periods = seperate(df)
    res = regression_analysis(file, periods)
    display(res)
    res_ = res[file[20:-4]]
    titles = ['Before', 'Between', 'After']
    res_LR = []
    res_NN = []
    res_SVM = []
    for i in range(3):
        cur_res = res_[titles[i]]
        res_LR.append(cur_res['LR'])
        res_NN.append(cur_res['NN'])
        res_SVM.append(cur_res['SVM'])
        
    df = pd.DataFrame({
        file[20:-4] : titles,
        'Linear Regression' : res_LR,
        'Neural Network' : res_NN,
        'SVM' : res_SVM
    }, columns = [file[20:-4], 'Linear Regression', 'Neural Network', 'SVM'])
    display(df)

