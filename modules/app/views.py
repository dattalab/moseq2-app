from flask import render_template, make_response, jsonify, send_from_directory, url_for
from app import app
import os
import logger

@app.route('/')
def index():
    ''' static files serve '''
    return render_template('index.html')

@app.route('/extract')
def extract():
    ''' static files serve '''
    return render_template('extract.html')

@app.route('/pca')
def pca():
    ''' static files serve '''
    return render_template('pca.html')

@app.route('/model')
def model():
    ''' static files serve '''
    return render_template('model.html')

@app.route('/viz')
def viz():
    ''' static files serve '''
    return render_template('viz.html')

@app.route('/<path:path>')
def static_proxy(path):
    """ static folder serve """
    file_name = path.split('/')[-1]
    dir_name = os.path.join('', '/'.join(path.split('/')[:-1]))
    return send_from_directory(dir_name, file_name)