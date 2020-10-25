# coding=utf-8           # -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import Api, Resource, reqparse
import random
from functools import reduce
from time import time
import logging
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import httplib2
import gspread
from google.oauth2.service_account import Credentials
import random
from config import CREDENTIALS_FILE, spreadsheetId
import json

def normailize_text(
        data,
        tok=RegexpTokenizer(r'\w[\w\/\-]+'),
        stemmers=[RussianStemmer(ignore_stopwords=True), PorterStemmer()]
):
    # tokenize text into words
    # sequentially apply all stemmers to tokenized words
    # join stemmed words back to sentences
    return [' '.join([reduce(lambda v,f: f.stem(v), stemmers, w) for w in tok.tokenize(line)])
            for line in data]
grid_search = joblib.load('gs_object.pkl')
app = Flask(__name__)
api = Api(app)
# Авторизуемся в Google Sheep
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_file(
    CREDENTIALS_FILE,
    scopes=scopes
)
service = gspread.authorize(credentials)
sh = service.open_by_url("https://docs.google.com/spreadsheets/d/" + spreadsheetId)
worksheep = sh.get_worksheet(0)


class Quote(Resource):
    def get(self, fg="Как активировать карточку"):
        test = pd.DataFrame({'Name':[fg]})
        test['Code'] = grid_search.predict(normailize_text(test['Name'].values.tolist()))
        print(test["Code"][0])
        cell = worksheep.find(str(test["Code"][0]))
        nomer_row, nomer_col = cell.row, cell.col
        print(nomer_row)
        values_list = worksheep.row_values(nomer_row)
        print(values_list[8:])
        results = ''
        l = 1
        for i in values_list[8:]:
            results=str(results + str(l) +". " + str(i) + "\n")
            l+=1 
        print(results)
        data = {
            'answer': results,
        }
        return data,200
        return "Quote not found", 404


api.add_resource(Quote, "/fgu", "/fgu/", "/fgu/<string:fg>")
if __name__ == '__main__':
    app.run(debug=True)
