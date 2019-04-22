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
    
#compute idf
def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
	result = {}
	for word in inv_idx:
	    DF = len(inv_idx[word])
	    if word == '.':
	        print(DF)
	    if DF >= min_df:
	        IDF = math.log((n_docs / (1 + DF)), 2)
	        if DF/n_docs <= max_df_ratio:
	            result[word] = IDF
	return result

#compute norms
def compute_doc_norms(index, idf, n_docs):
    result = np.zeros(n_docs)
    for word in index:
        for doc in index[word]:
            if word in idf:
            	result[doc] += idf[word] ** 2
    result = np.sqrt(result)
    return result

def index_search(query, index, idf, doc_norms):
	dic = {}
	tokens = tokenize(query.lower())
	for token in tokens:
	    if token in index:
	        for (doc_id) in index[token]:
	            if doc_id not in dic:
	                dic[doc_id] = tokens.count(token) * idf[token] * idf[token]
	            else :
	                dic[doc_id] += tokens.count(token)
	                
	#compute query norm
	q = 0
	for word in index:
	    if word in tokens:
	        q += ((tokens.count(word)) * idf[word]) ** 2
	q = q ** .5

	result = []
	for doc in dic:
	    doc_norm = doc_norms[doc]
	    score = dic[doc] / (doc_norm * q)
	    result.append((score, doc))

	result.sort(key = lambda x: x[0], reverse = True)
	return result



#returns a list of drinks that match the paramter
@app.route("/search-results/", methods=['GET', 'POST'])
def search():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
	data = json.load(open(json_url))

	#getting the query document
	print(request)
	query = request.args.get('query')
	print(query)

	n_docs = []
	#construct inverted index
	inverted_index = {}
	reverse_index_good_words = {}
	words = {}
	doc_index = 0
	for drink in data["drinks"]:
		tokens = tokenize(drink["description"])
		n_docs.append(drink["name"])
		for token in tokens:
			if token not in words:
				words[token] = 0
			words[token] += 1
			if token not in inverted_index:
				inverted_index[token] = []
			inverted_index[token].append(doc_index) #make a tuple
		doc_index += 1
	good_words = []
	index = 0
	for word in words:
	    if words[word] > 1:
	        good_words.append(word)
	        reverse_index_good_words[word] = index 
	        index += 1

	#compute idf
	idf = compute_idf(inverted_index, len(good_words))
	doc_norms = compute_doc_norms(inverted_index, idf, len(n_docs))
	results = index_search(query, inverted_index, idf, doc_norms)
	results = [n_docs[x[1]] for x in results]
	print(results)
	output = [x for x in data["drinks"] if x["name"] in results]
	return json.dumps(output)
# returns an array of good types
@app.route("/good-types/", methods=['GET', 'POST'])
def return_good_types():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
	data = json.load(open(json_url))

	#good types
	good_words = []
	words = {}
	for drink in data["drinks"]:
		tokens = tokenize(drink["description"])
		if tokens is not None:
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
