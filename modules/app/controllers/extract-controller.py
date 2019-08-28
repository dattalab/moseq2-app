from moseq2_extract.cli import *
from moseq2_batch.cli import *
import os
from flask import request, jsonify
from app import app#, mongo
import logger

@app.route('/generate-config', methods=['POST'])
def extract_generate_conf(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/gen-gridsearch-config', methods=['POST'])
def batch_gen_gridsearch_conf(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/extract-raw', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def extract_raw(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/extract-batch', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def batch_extract_raw(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/aggregate-extract-results', methods=['GET', 'POST', 'DELETE', 'PATCH'])
def batch_agg_extract_results(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/download-flip-file', methods=['GET'])
def extract_download_flip(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/find-roi', methods=['GET', 'POST', 'PATCH'])
def extract_find_roi(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/copy-slice', methods=['GET', 'POST', 'PATCH'])
def extract_copy_slice(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

