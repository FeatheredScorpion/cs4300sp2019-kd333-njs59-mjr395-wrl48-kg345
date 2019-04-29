# Gevent needed for sockets
from gevent import monkey

monkey.patch_all()

# Imports
import os
from flask import Flask, render_template, redirect, url_for, json, request
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
import sys
import pickle, re
from urllib.parse import urlparse
import math
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.config.from_object(os.environ["APP_SETTINGS"])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
db = SQLAlchemy(app)

# Import + Register Blueprints
from app.accounts import accounts as accounts

app.register_blueprint(accounts)
from app.irsystem import irsystem as irsystem

app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

# treebank tokenizer for boolean search
treebank_tokenizer = TreebankWordTokenizer()

# inverted index
inverted_index = {}
# good types
good_types = []

reverse_index_good_words = {}

results = []
#range or results to display - page - 5 : 5
page = 10
#document numbers
n_doc = []

words = {}



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


# compute idf
def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    result = {}
    for word in inv_idx:
        DF = len(inv_idx[word])
        if DF >= min_df:
            IDF = math.log((n_docs / (1 + DF)), 2)
            if DF / n_docs <= max_df_ratio:
                result[word] = IDF
    return result


# compute norms
def compute_doc_norms(index, idf, n_docs):
    result = np.zeros(n_docs)
    for word in index:
        for (doc, category) in index[word]:
            if word in idf:
                result[doc] += idf[word] ** 2
    result = np.sqrt(result) 
    return result


def index_search(query, index, idf, doc_norms):
    category_weights = {
    'review': 1.3,
    'title': 1.6,
    'tags': 1.3,
    'description': 1,
    'categories' : 1.3
    }
    dic = {}
    tokens = tokenize(query.lower())
    for token in tokens:
        if token in index:
            for (doc_id, category) in index[token]:
                if doc_id not in dic:
                    if token in idf:
                        idf_token = idf[token]
                        dic[doc_id] = category_weights[category] * float(tokens.count(token) * idf_token * idf_token)
                else:
                    if token in idf:
                        idf_token = idf[token]
                        dic[doc_id] += category_weights[category] * float(tokens.count(token) * idf_token * idf_token)
    # compute query norm
    q = 0
    for word in index:
        if word in tokens:
            if token in idf:
                idf_token = idf[token]
                q += ((tokens.count(word)) * idf_token ** 2)
    q = q ** .5

    result = []
    i = 0
    reverse_doc_index = {}
    for doc in dic:
    	reverse_doc_index[doc] = i
    	doc_norm = doc_norms[doc]
    	score = (1 + dic[doc]) / (doc_norm * q + 1)
    	result.append((score, doc))
    	i = i + 1

    return (result, reverse_doc_index)

def create_inverted_index():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks-with-related-and-tags.json")
    data = json.load(open(json_url))

    global inverted_index
    doc_index = 0
    global n_docs
    n_docs = []
    global words
    words = {}

    for drink in data["drinks"]:
        n_docs.append(drink["name"])
        tokens = []
        for review in drink["reviews"]:
        	tokens = tokens + tokenize(review["body"])
        for token in tokens:
        	if token not in words:
        		words[token] = 0
        	words[token] += 1
        	if token not in inverted_index:
        		inverted_index[token] = []
        	inverted_index[token].append((doc_index, 'review'))
        tokens = tokenize(drink["name"])     
        for token in tokens:
        	if token not in words:
        		words[token] = 0
        	words[token] += 1
        	if token not in inverted_index:
        		inverted_index[token] = []
        	inverted_index[token].append((doc_index, 'title'))   
        tokens = tokenize(drink["description"])
        for token in tokens:
        	if token not in words:
        		words[token] = 0
        	words[token] += 1
        	if token not in inverted_index:
        		inverted_index[token] = []
        	inverted_index[token].append((doc_index, 'description'))
        tokens = []
        for tag in drink["tags"]:
        	tokens = tokens + tokenize(tag)
        for token in tokens:
        	if token not in words:
        		words[token] = 0
        	words[token] += 1
        	if token not in inverted_index:
        		inverted_index[token] = []
        	inverted_index[token].append((doc_index, 'tags'))
        tokens = []
        for tag in drink["categories"]:
        	tokens = tokens + tokenize(tag)
        for token in tokens:
        	if token not in words:
        		words[token] = 0
        	words[token] += 1
        	if token not in inverted_index:
        		inverted_index[token] = []
        	inverted_index[token].append((doc_index, 'categories'))
        doc_index += 1
    return json.dumps([])

# returns a list of drinks that match the paramter
@app.route("/search-results/", methods=['GET', 'POST'])
def search():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks-with-related-and-tags.json")
    data = json.load(open(json_url))

    # getting the query document
    query = request.args.get('search')
    ingredients = tokenize(request.args.get('ingredients'))
    r = []
    if query[0] == '"' and query[-1] == '"':
    	stripped = query[1:-1]
    	for drink in data["drinks"]:
    		if drink["name"].lower() == stripped.lower():
    			r.append(drink)
    else:
	    # construct inverted index
	    global inverted_index
	    # inverted_index = {}
	    global reverse_index_good_words
	    reverse_index_good_words = {}
	    global results
	    results = []
	    global page
	    page = 10
	    global n_docs

	    if not inverted_index:
	    	create_inverted_index()

	    results = []


	    good_words = []
	    index = 0
	    for word in words:
	        if words[word] > 1:
	            good_words.append(word)
	            reverse_index_good_words[word] = index
	            index += 1
	    # compute idf
	    idf = compute_idf(inverted_index, len(good_words))
	    doc_norms = compute_doc_norms(inverted_index, idf, len(n_docs))
	    (results, reverse_doc_index) = index_search(query, inverted_index, idf, doc_norms)
	    #if no query, search based on ingredients

	    if(len(query) == 0):
	        for i in range(len(n_docs)):
	            reverse_doc_index[i] = i
	            results.append((1, i))
	    drink_index = 0
	    for drink in data["drinks"]:
	        drink_ingredients = []
	        for i in drink["ingredients"]:
	            drink_ingredients += [tokenize(x) for x in drink["ingredients"]]
	            drink_ingredients = [item for sublist in drink_ingredients for item in sublist]
	        for ingredient in drink_ingredients:
	            if ingredient in ingredients and drink_index in reverse_doc_index:
	                results[reverse_doc_index[drink_index]] = (results[reverse_doc_index[drink_index]][0] * 1.8, results[reverse_doc_index[drink_index]][1])
	        drink_index += 1 

	    results.sort(key = lambda x: x[0], reverse = True)
	    results = [n_docs[x[1]] for x in results]
	    #results = [x for x in results if x in drinks_w_ingredients]
	    print(results)
	    results = results
	    r = []
	    for name in results[:page]:
	    	for drink in data["drinks"]:
	    		if drink["name"] == name:
	    			r.append(drink)
	    output = [x for x in data["drinks"] if x["name"] in results]
    return json.dumps(r)

@app.route("/load-more/", methods=['GET', 'POST'])
def load_more_results():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks-with-related-and-tags.json")
    data = json.load(open(json_url))
    global page
    r = []
    re = results[page:page + 5]
    for name in re:
    	for drink in data["drinks"]:
    		if drink["name"] == name:
    			r.append(drink)
    output = [x for x in data["drinks"] if x["name"] in re]
    page = page + 5
    return json.dumps(r)


# returns an array of good types
@app.route("/good-types/", methods=['GET', 'POST'])
def return_good_types():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks-with-related-and-tags.json")
    data = json.load(open(json_url))

    # good types
    words = {}
    for drink in data["drinks"]:
        tokens = tokenize(drink["description"])
        for review in drink["reviews"]:
            tokens = tokens + tokenize(review["body"])
        if tokens is not None:
            for token in tokens:
                if token not in words:
                    words[token] = 0
                words[token] += 1
    global good_types
    good_types = []
    for word in words:
        if words[word] > 1:
            good_types.append(word)
    return json.dumps(good_types)

@app.route("/autocomplete-types/", methods=['GET', 'POST'])
def return_autocomplete_types():
    global good_types
    if (len(good_types) < 100):
        return_good_types()



# returns an array of good types from ingredients
@app.route("/good-ingredients/", methods=['GET', 'POST'])
def return_ingredients():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static", "drinks-with-related-and-tags.json")
    data = json.load(open(json_url))

    # good types
    good_words = []
    words = {}
    for drink in data["drinks"]:
        tokens = []
        for item in drink["ingredients"]:
        	tokens += tokenize(item)
        if tokens != []:
            for token in tokens:
                if token not in words:
                    words[token] = 0
                words[token] += 1
    good_words = []
    for word in words:
        if words[word] > 1:
            good_words.append(word)
    return json.dumps(good_words)

create_inverted_index()

