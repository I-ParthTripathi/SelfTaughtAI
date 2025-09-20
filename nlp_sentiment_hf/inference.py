from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
clf = pipeline('sentiment-analysis')

@app.route('/sentiment', methods=['POST'])
def sentiment():
    text = request.json.get('text','')
    out = clf(text)
    return jsonify(out)