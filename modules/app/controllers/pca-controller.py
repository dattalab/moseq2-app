from moseq2_pca.cli import *
import os
from flask import request, jsonify
from app import app#, mongo
import logger

@app.route('/train-pca', methods=['GET', 'POST', 'PATCH'])
def pca_train(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/apply-pca', methods=['GET', 'POST', 'PATCH'])
def pca_apply(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/add-groups', methods=['GET'])
def pca_add_groups(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/compute-changepoints', methods=['GET', 'POST', 'PATCH'])
def pca_compute_cps(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/clip-scores', methods=['GET', 'POST', 'PATCH'])
def pca_clip_scores(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

