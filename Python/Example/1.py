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

def read_input_data_set(filename):
    # read CSV into a DF
    df = pd.read_csv("test.csv")
    # make DF bigger: 3 rows -> 15 rows
    return pd.concat([df], ignore_index=True)

# tokenize and stem text
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



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # read CSV into a DF
    df = pd.read_csv("test.csv")
    print('Input Data Set:')
    print(df)
    print()

    # word tokenizer
    tok = RegexpTokenizer(r'\w[\w\/\-]+')
    en = PorterStemmer()
    ru = RussianStemmer(ignore_stopwords=True)

    data = normailize_text(df['Name'].values.tolist(), tok=tok, stemmers=[ru,en])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        #'tfidf__max_df': (0.5, 0.75, 1.0),
        'tfidf__max_features': (None, 10000, 50000, 100000),
        #'tfidf__stop_words': ['russian','english'],
        'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'clf__alpha': np.logspace(-7, 2, 10),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__max_iter': (1000, 5000, 10000, 100000),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=3, verbose=1)

    # train model
    t0 = time()
    grid_search.fit(data, df['Code'])
    print("done in %0.3fs" % (time() - t0))
    print()

    print('best parameters:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print()

    best_parameters = grid_search.best_estimator_.get_params()
    joblib.dump(grid_search, 'gs_object.pkl')
    test = pd.DataFrame({'Name':['Как активировать карту', 'заблокировать карту', 'платежи по карте']})
    print()    

    # predict codes
    test['Code'] = grid_search.predict(normailize_text(test['Name'].values.tolist()))

    # print results
    print('test data set after prediction:')
    print(test)