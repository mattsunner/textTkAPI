from flask import Flask, render_template, request, jsonify
from textTkAPI import expand_contractions, lemmatize_sentence, tokenizer

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"message": "Success!"})


@app.route('/tokenize', methods=['GET', 'POST'])
def tokenize():

    if request.method == 'POST':
        corpus = request.form['corpus']
        tokens = tokenizer(corpus)
        return jsonify({"tokens": tokens})
    else:
        return jsonify({"Error": "This route only accpets POST methods. Please try again."})


@app.route('/lemmatize', methods=['POST'])
def lemmatize():
    if request.method == 'POST':
        corpus = request.form['corpus']
        lemms = lemmatize_sentence(corpus)
        return jsonify({"Lemms": lemms})
    else:
        return jsonify({"Error": "This route only accpets POST methods. Please try again."})


if __name__ == '__main__':
    app.run(debug=True)
