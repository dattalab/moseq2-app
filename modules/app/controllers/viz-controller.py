from moseq2_viz.cli import *
import os
from flask import request, jsonify
from app import app#, mongo
import logger

@app.route('/viz-add-group', methods=['GET'])
def viz_add_group(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/generate-viz-index', methods=['GET', 'POST', 'PATCH'])
def viz_gen_index(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/make-crowd-movies', methods=['GET', 'POST', 'PATCH'])
def viz_make_crowd_movies(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-scalar-summary', methods=['GET', 'POST', 'PATCH'])
def viz_plot_scalar_summary(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-transition-graph', methods=['GET', 'POST', 'PATCH'])
def viz_plot_trans_graph(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-usages', methods=['GET', 'POST', 'PATCH'])
def viz_plot_usages(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400