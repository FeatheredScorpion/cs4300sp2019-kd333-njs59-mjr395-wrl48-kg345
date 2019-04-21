# Gevent needed for sockets
from gevent import monkey

monkey.patch_all()

# Imports
import os
from flask import Flask, render_template, redirect, url_for, json
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
import sys
import pickle, re
from nltk.tokenize import TreebankWordTokenizer

# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.config.from_object(os.environ["APP_SETTINGS"])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
db = SQLAlchemy(app)

# inverted index
inverted_index = {}
# good types
good_words = []
reverse_index_good_words = {}

# Import + Register Blueprints
from app.accounts import accounts as accounts

app.register_blueprint(accounts)
from app.irsystem import irsystem as irsystem

app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

#treebank tokenizer for boolean search
treebank_tokenizer = TreebankWordTokenizer()


# HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


def tokenize(text):
    """Returns a list of tokens from an input string.
    
    Params: {text: string}
    Returns: List
    """
    return [x for x in re.findall(r"[a-z]*", text.lower()) if x != ""]


#

# precompute inverted_index  and good types
@app.route("/hello/", methods=['GET', 'POST'])
def hello():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
    data = json.load(open(json_url))

    # create good types matrix
    words = {}
    for drink in data["drinks"]:
        tokens = tokenize(drink["description"])
        for token in tokens:
            if token not in words:
                words[token] = 0
            words[token] += 1
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(drink["name"])
    print(inverted_index)
    good_words = []
    index = 0
    for word in words:
        if words[word] > 1:
            good_words.append(word)
            reverse_index_good_words[word] = index
            index += 1
    # construct inverted index
    # print(good_words)
    return redirect('/')


# returns an array of good types
@app.route("/good-types/", methods=['GET', 'POST'])
def return_good_types():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
    data = json.load(open(json_url))

    # good types
    good_words = []
    words = {}
    for drink in data["drinks"]:
        tokens = tokenize(drink["name"])
        for token in tokens:
            if token not in words:
                words[token] = 0
            words[token] += 1
    good_words = []
    for word in words:
        if words[word] > 1:
            good_words.append(word)
    return json.dumps(good_words)

# returns a list of resulting elements that satisfies the boolean search
@app.route("/booleanSearch/", methods=['GET', 'POST'])
def boolean_search(query, index=inverted_index, tokenizer=treebank_tokenizer):
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
    data = json.load(open(json_url))

    ans = []
    temp1 = set()
    temp2 = set()
    tokenized = tokenizer.tokenize(query.lower())
    count = 0
    for x in tokenized:
        temp2 = set([a[0] for a in index[x]])
        if(count == 0):
            ans = list(temp2)
            count+=1
            continue
        if(len(ans) == 0):
            return ans
        if(len(x) == 1 and x in string.punctuation):
            count+=1
            continue
        else:
            temp1 = set(ans)
            ans = list(temp2.intersection(temp1))
    return ans