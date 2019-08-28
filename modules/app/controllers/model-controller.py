from moseq2_model.cli import *
import os
from flask import request, jsonify
from app import app#, mongo
import logger

@app.route('/learn-model', methods=['GET', 'POST', 'PATCH'])
def model_learn(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/learn-model-parameter-scan', methods=['GET', 'POST', 'PATCH'])
def batch_learn_model(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/aggregate-modeling-results', methods=['GET', 'POST', 'PATCH'])
def batch_agg_modeling_res(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/count-frames', methods=['GET', 'POST', 'PATCH'])
def model_count_frames(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400