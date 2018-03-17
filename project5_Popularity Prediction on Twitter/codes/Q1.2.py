
# coding: utf-8

# In[146]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
import datetime, time
import pytz


# define paths
files = ['extracted_data/Q1.1_#gohawks.csv', 'extracted_data/Q1.1_#gopatriots.csv', 'extracted_data/Q1.1_#nfl.csv', 'extracted_data/Q1.1_#patriots.csv', 'extracted_data/Q1.1_#sb49.csv', 'extracted_data/Q1.1_#superbowl.csv']

# calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# load and process data from each hashtag file
def load_and_process(file):
    # process and groupby data
    data = pd.read_csv(file)
    data.columns = ['tweet', 'timestamp', 'followers', 'retweets']
    # timestamp -> time zone
    date = []
    time = []
    for timestamp in data['timestamp']:
        pst_tz = pytz.timezone('US/Pacific')
        timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
        date_split = timestamp[0:10].split('-')
        date.append(int(date_split[0]+date_split[1]+date_split[2]))
        time.append(int(timestamp[11:13]))
    data.insert(1, 'date', date)
    data.insert(2, 'time', time)
    data.insert(3, 'followers_max', data['followers'])
    data.drop('timestamp', 1, inplace = True)
    df = data.groupby(['date', 'time']).agg({'date' : np.max, 'time' : np.max, 'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'followers_max' : np.max})
    
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
                app_rows.append({'tweet':0,'date':pre_date,'time':pre_hour-24,'followers_max':0,'followers':0,'retweets':0})
            else:
                app_rows.append({'tweet':0,'date':pre_date,'time':pre_hour,'followers_max':0,'followers':0,'retweets':0})
            hour_diff = cur_hour - pre_hour
    for row in app_rows:
        data = data.append(row, ignore_index=True)
    df = data.groupby(['date', 'time']).agg({'time' : np.max, 'tweet' : np.sum, 'retweets' : np.sum, 'followers' : np.sum, 'followers_max' : np.max})
    df.to_csv('extracted_data/Q1.2_hourly_'+file[20:-4]+'.csv', index=False)
    display(df)
    return df

# train and fit linear regression model
def regression_analysis(file, df):
    input_arr = []
    for index in df.index:
        input_arr.append(df.loc[index, 'time':'followers_max'].values)
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
    
    # plot regression fitting results
    fig = plt.figure(figsize=(15,9))
    fig = sm.graphics.plot_partregress_grid(results, fig=fig)
    fig.show()
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
    print ('==========================================================================================================')
    print ()

# linear regression model on specified hashtag
def regression_model(file):
    df = load_and_process(file)
    regression_analysis(file, df)

# linear regression model on each hashtag
for file in files:
    regression_model(file)

