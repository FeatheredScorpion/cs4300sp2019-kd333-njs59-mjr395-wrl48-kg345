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



# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.config.from_object(os.environ["APP_SETTINGS"])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
db = SQLAlchemy(app)

#inverted index
inverted_index = {}
#good types
good_words = []
reverse_index_good_words = {}

# Import + Register Blueprints
from app.accounts import accounts as accounts
app.register_blueprint(accounts)
from app.irsystem import irsystem as irsystem
app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

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

#precompute inverted_index  and good types
@app.route("/hello/", methods=['GET', 'POST'])
def hello():
	SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(SITE_ROOT, "static", "drinks.json")
	data = json.load(open(json_url))

	#create good types matrix
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
	#construct inverted index
	#print(good_words)
	return redirect('/')





