import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer, f1_score, precision_score, recall_score


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    # load data from database
    sql_engine = create_engine('sqlite:///'+database_filepath, echo=False)
    connection = sql_engine.raw_connection()  
    table_name = str(sql_engine.table_names()[0])
    
    print(sql_engine.table_names())

    df = pd.read_sql("SELECT * FROM '{}'".format(table_name),con=connection)
    cols = list(set(df.columns)-set(df[['id','message','original','genre']]))
    
    df = df[(df.related!=2) & (df[cols].sum(axis=1)!=0)]

    X = df['message']  #remove original
    Y = df.drop(columns=['id','message','original','genre'])
    
    return X, Y, Y.columns

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
        text = text.replace(ur,'')
        
    detected_urls = re.findall(ANY_URL_REGEX, text)
    for url in detected_urls:
        # print('url:', url)
        text = text.replace(url,'')

    detected_emails = re.findall(EMAIL_REGEX, text)
    for email in detected_emails:
        # print('email:', email)
        text = text.replace(email[0], '')
    
    pattern = re.compile(r'[^a-zA-Z]') # re.compile(r'[^a-zA-Z0-9]') remove numbers
    stopword_list = stopwords.words('english')

    for stop_word in stopword_list:
        
        if(stop_word in text):
             text.replace(stop_word,'')
    
    text = re.sub(pattern,' ',text)
    
    tokens = word_tokenize(text.lower())
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        if((tok not in stopword_list) and len(tok)>2):      

            clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok.strip()),pos='v')
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    pipeline works better than any with xgboost
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(), n_jobs=-1))
        ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    count= 0
    for col in category_names:
        print(col, classification_report(
            Y_test[col].values,
            y_pred[:,count]))
        count += 1
    print(classification_report(Y_test,y_pred,target_names=list(category_names)))
    count = 0
    for col in category_names:
        acc = accuracy_score(Y_test[col],y_pred[:,count])
        f1 = f1_score(Y_test[col],y_pred[:,count],average='micro')
        prec = precision_score(Y_test[col],y_pred[:,count],average='micro')
        recall = recall_score(Y_test[col],y_pred[:,count],average='micro')
        print('{0:.3f}, {0:.3f},  {0:.3f}, {0:.3f}    '.format(acc,f1),col)
        count += 1
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()