import json
import plotly
import pandas as pd
import os
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
# import spacy
# import en_core_web_sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

nltk.download(['stopwords', 'punkt', 'wordnet'])

app = Flask(__name__)


def tokenize(text):
    """
    The regex of URL and Email now correctly find and remove them
    """
    FIRST_URL_REGEX = re.compile(r"""http\s[a-z0-9](?:[a-z0-9-.]*[a-z0-9].)?\s[a-z0-9](?:[a-z0-9-?=]*[a-z0-9?=])?""")
    ANY_URL_REGEX = re.compile(r"""(?i)\b((?:https?\s?:\s?(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""")
    EMAIL_REGEX = re.compile(r"""(?i)([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`""{|}~-]+)*(@)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)""")
    stopword_list = stopwords.words('english')

    # all cases of pattern save in string for each case
    first_detected_url = re.findall(FIRST_URL_REGEX, text)
    for ur in first_detected_url:
        # print('ur', ur)
        text = text.replace(ur, "")
        
    detected_urls = re.findall(ANY_URL_REGEX, text)
    for url in detected_urls:
        # print('url:', url)
        text = text.replace(url, "")

    detected_emails = re.findall(EMAIL_REGEX, text)
    for email in detected_emails:
        # print('email:', email)
        text = text.replace(email[0], "")

    # remove numbers
    pattern = re.compile(r'[^a-zA-Z]')
    stopword_list = stopwords.words('english')

    for stop_word in stopword_list:
        
        if stop_word in text:
            text.replace(stop_word, '')
    
    text = re.sub(pattern, ' ', text)
    
    tokens = word_tokenize(text.lower())
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        if (tok not in stopword_list) and (len(tok) > 2):

            clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok.strip()), pos='v')
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
connection = engine.raw_connection()

df = pd.read_sql("SELECT * FROM '{}'".format('Data'), con=connection)
df.drop('child_alone', axis=1, inplace=True)
# load model
# model = joblib.load("../models/fair_model__ada_01.pkl")
model = pickle.load(open('../models/fair_model__ada_02.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    # genre and aid_related status
    aid_rel1 = df[df['aid_related'] == 1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related'] == 0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)

    # let's calculate distribution of classes with 1
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()/len(df)

    # sorting values in ascending
    class_distr1 = class_distr1.sort_values(ascending=False)

    # series of values that have 0 in classes
    class_distr0 = (class_distr1 - 1) * -1
    class_name = list(class_distr1.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name='Aid related'

                ),
                Bar(
                    x=genre_names,
                    y=aid_rel0,
                    name='Aid not related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=class_name,
                    y=class_distr1,
                    name='Class = 1'
                    # orientation = 'h'
                ),
                Bar(
                    x=class_name,
                    y=class_distr0,
                    name='Class = 0',
                    marker=dict(
                            color='rgb(212, 228, 247)'
                                )
                    # orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
                    # 'tickangle': -45
                },
                'barmode': 'stack'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
