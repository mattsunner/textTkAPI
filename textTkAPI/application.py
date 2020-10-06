from flask import Flask, render_template, request
import nltk
import pandas as pd
import numpy as np
import re
from nltk import pos_tag, pos_tag_sents
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams

from textProcessing import expand_contractions, lemmatize_sentence

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tokenize', methods=['GET', 'POST'])
def tokenize():

    corpus = request.form.to_dict()
    print(corpus)
    dfTokens = pd.DataFrame(corpus, index=[0])
    dfTokens.columns = ['token']

    dfTokens['token'] = dfTokens['token'].apply(
        lambda x: expand_contractions(x))

    dfTokens['lemmatized'] = dfTokens['token'].apply(
        lambda x: lemmatize_sentence(x))

    stopword_list = stopwords.words('english')

    dfTokens['lemmatized'] = dfTokens['lemmatized'].str.lower()
    dfTokens['lemmatized'] = dfTokens['lemmatized'].str.replace(
        '[^\w\s]', '').apply(word_tokenize)
    dfTokens['lemmatized'] = dfTokens['lemmatized'].apply(
        lambda x: [item for item in x if item not in stopword_list])

    dfTokens = dfTokens.drop(columns=['token'])
    print(dfTokens)
    tokensraw = dfTokens.values.tolist()
    tokens = tokensraw

    return render_template('tokenize.html', tokens=tokens)


if __name__ == '__main__':
    app.run(debug=True)
