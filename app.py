from flask import Flask, render_template, request, jsonify
from textTkAPI import expand_contractions, lemmatize_sentence, tokenizer

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"message": "Success!"})


@app.route('/tokenize', methods=['GET', 'POST'])
def tokenize():

    if request.method == 'POST':
        corpus = request.form['sentence']
        tokens = tokenizer(corpus)
        return jsonify({"tokens": tokens})
    else:
        return jsonify({"message": "This is the GET route /tokenize"})


if __name__ == '__main__':
    app.run(debug=True)
