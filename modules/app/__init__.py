''' flask app '''
import os
import json
import datetime
from bson.objectid import ObjectId
from flask import Flask
from flask_pymongo import PyMongo

class JSONEnconder(json.JSONEncoder):
    ''' extend json-encoder class '''

    def default(self, o):
        if isinstance(o, ObjectId): # used to store '_id'
            return str(o)
        if isinstance(o, datetime.datetime): # used to sore 'time-stamp'
            return str(o)
        return json.JSONEncoder.default(self, o)

# create the flask object
app = Flask(__name__)

# add mongo url to flask config, so that flask_pymongo can use it to make connections
app.config['MONGO_URI'] = os.environ.get('DB')
mongo = PyMongo(app)

# use the modified encoder class to handle ObjectId & datetime object while jsonifying the response
app.json_encoder = JSONEnconder

from app.controllers import *