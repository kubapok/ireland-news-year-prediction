import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
import random
import pickle

def tokenizer_space(text):
    return text.split(' ')



def run():

    # LOADING DATA
    train_text = [a.rstrip('\n') for a in open('../train/in.tsv','r')]
    dev_text = [a.rstrip('\n') for a in open('../dev-0/in.tsv','r')]
    test_text = [a.rstrip('\n') for a in open('../test-A/in.tsv','r')]
    global lowest

    train_year = [float(a.rstrip('\n')) for a in open('../train/expected.tsv','r')]
    dev_year = [float(a.rstrip('\n')) for a in open('../dev-0/expected.tsv','r')]

    max_year = max(train_year)
    min_year = min(train_year)

    tfidf = TfidfVectorizer()
    #tfidf = HashingVectorizer()
    train_text_vectorized = tfidf.fit_transform(train_text)
    pickle.dump(train_text_vectorized, open('text_train_tfidf_all.pickle','wb'))
    pickle.dump(tfidf, open('tfidf_all.pickle','wb'))
    train_text_vectorized = pickle.load(open('text_train_tfidf_all.pickle','rb'))
    tfidf = pickle.load(open('tfidf_all.pickle','rb'))

    dev_text_vectorized = tfidf.transform(dev_text)
    test_text_vectorized = tfidf.transform(test_text)

    # MODELLING
    lr = LinearRegression( n_jobs=10)
    #xgb = XGBRegressor(n_jobs=8)
    #xgb_1000 = XGBRegressor(n_estimators=1000,n_jobs=8)
    #xgb_5000 = XGBRegressor(n_estimators=5000,n_jobs=8)
    lr.fit(train_text_vectorized, train_year)
    #xgb.fit(text, year)
    #xgb_1000.fit(text, year)
    #xgb_5000.fit(text, year)


    ##################
    # DEV PREDICTIONS
    predictions_lr = lr.predict(dev_text_vectorized)
    predictions_lr = np.minimum(predictions_lr, max_year)
    predictions_lr = np.maximum(predictions_lr, min_year)
    print('dev-0 RMSE')
    print(np.sqrt(sklearn.metrics.mean_squared_error(predictions_lr, dev_year)))
    print('dev-0 MAE')
    print(sklearn.metrics.mean_absolute_error(predictions_lr, dev_year))

    f = open('../dev-0/out.tsv','w')
    for i in predictions_lr:
        f.write(str(i) + '\n')
    f.close()

    ##################
    # TEST PREDICTIONS
    predictions_lr = lr.predict(test_text_vectorized)
    predictions_lr = np.minimum(predictions_lr, max_year)
    predictions_lr = np.maximum(predictions_lr, min_year)

    f = open('../test-A/out.tsv','w')
    for i in predictions_lr:
        f.write(str(i) + '\n')
    f.close()


run()
