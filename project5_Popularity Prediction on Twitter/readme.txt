============================ EE219 Project 5 ==============================
Jui Chang 
Wenyang Zhu 
Xiaohan Wang 
Yang Tang 


======================= Environment & Dependencies ========================
Imported libraries:

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import math
from pytz import timezone
import nltk
import calendar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

Functions:

plot_histogram(timestamp, file) for Q1.1 
1. Input: 
	timestamp: citation_date
	file: name of raw hashtag

cal_statistics(file) for Q1.1
1. Input:
	file: name of raw hashtag

rmse(predictions, targets) for all RMSE calculating
1. Input:
	predictions: predicted values
	targets: true values
2. Output: RMSE value

load_and_process(file) for all preprocessing of files
1. Input:
	file: name of raw hashtag
2. Output: DataFrame of the extracted data

regression_analysis(file, df) for all regression training and fitting
1. Input:
	file: name of raw hashtag
	df: extracted data frame

regression_model(file) for all hashtag to do regression analysis
1. Input:
	file: name of raw hashtag

plot_top3_features(file, df, output_predicted) for Q1.3
1. Input:
	file: name of raw hashtag
	df: extracted data frame
	output_predicted: predicted number of tweets

seperate(df) for Q1.4 & Q1.5
1. Input:
	df: extracted hourly data frame

three_models_ten_folds_errors(input_arr, output_arr) for Q1.4
1. Input: 
	input_arr: total data (train and test) of input
	output_arr: total data (train and test) of output
2. Output: average absolute error between true and predicted values

fit_predict(model, train_in, train_out, test_in) for Q1.4
1. Input:
	model: the name of the regression model
	train_in: input of train dataset
	train_out: output of train dataset
	test_in: input of test dataset
2. Output: predicted output of test dataset

regression_model_for_periods(periods) for Q1.5
1. Input:
	periods: period (before, between and after) of current data
2. Output: a period-model dict recording parameters of each model

predict_on_test_data(file, period_model, df) for Q1.5
1. Input:
	file: name of hashtag
	period_model: a dict recording parameters of each model
	df: data frame of the test dataset
2. Output: the predicted output of the test dataset

match(location) for Q2
1. Input: 
	location: the extracted data from raw hashtag dataset
2. Output: the label (1, -1, 0) of the location

trim_and_stem(data_list) for Q2
1. Input: 
	data_list: the data which needs to be trimmed and stemmed

plot_roc_curve(method, fpr, tpr, auc) for Q2
1. Input:
	method: the name of the classification method
	fpr, tpr, auc: the parameters of ROC curve

plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.YlOrBr) for Q2
1. Input:
	cm: true values and predicted values of test dataset
	classes: the binary classes of classification

clean_text_words(text) for Q3
1. Input: 
	a line of text
2. Output: 
	the clean tokenized word: delete url, hastag, symbol, stopwords, and stemmed 	words

=============================== Instruction ===============================
Each .py file solves one question and can be run seperately to get results
in console output.
	- Q1.1.py for Question 1.1
	- Q1.2.py for Question 1.2
	- Q1.3.py for Question 1.3
	- Q1.4.py for Question 1.4
	- Q1.5.py for Question 1.5
	- Q2.py for Question 2
	- Q3.py for Question 3

- all .ipynb files can be run directly on Jupyter Notebook and get the 
results below.

- extracted_data file contains part of our extracted data (.csv) in all questions.


============================= Folder Content ==============================
- codes
	- Q1.1.py
	- Q1.2.py
	- Q1.3.py
	- Q1.4.py
	- Q1.5.py
	- Q2.py
	- Q3.py	

	- Q1.1.ipynb
	- Q1.2.ipynb
	- Q1.3.ipynb
	- Q1.4.ipynb
	- Q1.5.ipynb
	- Q2.ipynb
	

- readme.txt
- report.pdf

