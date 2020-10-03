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
    contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not",
                         "can't": "cannot", "can't've": "cannot have",
                         "'cause": "because", "could've": "could have", "couldn't": "could not",
                         "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                         "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                         "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                         "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                         "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                         "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                         "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
                         "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                         "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                         "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                         "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                         "mustn't've": "must not have", "needn't": "need not",
                         "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                         "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                         "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                         "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                         "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                         "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                         "there'd've": "there would have", "they'd": "they would",
                         "they'd've": "they would have", "they'll": "they will",
                         "they'll've": "they will have", "they're": "they are", "they've": "they have",
                         "to've": "to have", "wasn't": "was not", "we'd": "we would",
                         "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                         "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                         "what'll've": "what will have", "what're": "what are", "what've": "what have",
                         "when've": "when have", "where'd": "where did", "where've": "where have",
                         "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                         "why've": "why have", "will've": "will have", "won't": "will not",
                         "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                         "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                         "y'all'd've": "you all would have", "y'all're": "you all are",
                         "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                         "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                         "you've": "you have"}

    contractions = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(text, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions.sub(replace, text)

    corpus = request.form.to_dict()
    print(corpus)
    dfTokens = pd.DataFrame(corpus, index=[0])
    dfTokens.columns = ['token']

    dfTokens['token'] = dfTokens['token'].apply(
        lambda x: expand_contractions(x))

    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    wordnet_lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)


    dfTokens['lemmatized'] = dfTokens['token'].apply(lambda x: lemmatize_sentence(x))

    stopword_list = stopwords.words('english')

    dfTokens['lemmatized'] = dfTokens['lemmatized'].str.lower()
    dfTokens['lemmatized'] = dfTokens['lemmatized'].str.replace('[^\w\s]','').apply(word_tokenize)
    dfTokens['lemmatized'] = dfTokens['lemmatized'].apply(lambda x: [item for item in x if item not in stopword_list])

    dfTokens = dfTokens.drop(columns=['token'])
    print(dfTokens)
    tokensraw = dfTokens.values.tolist()
    tokens = tokensraw

    return render_template('tokenize.html', tokens=tokens)


if __name__ == '__main__':
    app.run(debug=True)
