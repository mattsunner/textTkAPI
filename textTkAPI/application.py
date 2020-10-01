from flask import Flask, render_template, request
import nltk


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tokenize', methods=['GET', 'POST'])
def tokenize():
    sentence = request.form['tokens']
    tokens = nltk.word_tokenize(sentence)
    return render_template('tokenize.html', tokens=tokens)


if __name__ == '__main__':
    app.run(debug=True)
