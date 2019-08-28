''' index file for REST APIs using Flask '''
import os
import sys
import requests
from flask import jsonify, request, make_response, send_from_directory, render_template

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
os.environ.update({'ROOT_PATH': ROOT_PATH})
sys.path.append(os.path.join(ROOT_PATH, 'modules'))

#import logger
from app import app
from app.views import *

'''
# create a logger object to log the info and debug
LOG = logger.get_root_logger(os.environ.get(
    'ROOT_LOGGER', 'root'), filename=os.path.join(ROOT_PATH, 'output.log'))
'''
# Port variable to run the server on
PORT = os.environ.get('PORT')

@app.errorhandler(404)
def not_found(error):
    ''' error handler '''
    #LOG.error(error)
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    #LOG.info('running environment %s', os.environ.get('ENV'))
    app.config['DEBUG'] = os.environ.get('ENV') == 'development' # Debug mode if development env
    app.run(host='0.0.0.0', port=int(4000)) # Run the app