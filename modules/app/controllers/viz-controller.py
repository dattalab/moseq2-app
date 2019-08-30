from moseq2_viz.cli import *
import os
from flask import request, jsonify
from app import app, data_path#, mongo
import logger

@app.route('/viz-add-group', methods=['GET'])
def viz_add_group(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/generate-viz-index', methods=['GET', 'POST', 'PATCH'])
def viz_gen_index(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd + data_path

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz generate-index -i {cwd1} -p {cwd1}_pca/pca_scores.h5 -o {cwd1}moseq2-index.yaml')

            if os.path.exists(cwd1 + 'moseq2-index.yaml'):
                return jsonify({'ok': True, 'message': 'Index file successfully generated!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/make-crowd-movies', methods=['GET', 'POST', 'PATCH'])
def viz_make_crowd_movies(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        output_path = cwd1 + 'crowd_movies/'
        index_path = cwd1+'moseq2-index.yaml'
        model_path = cwd1+'model.p'

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            print(index_path, model_path)
            os.system(f'moseq2-viz make-crowd-movies -o {output_path} {index_path} {model_path}')

            if os.path.exists(cwd1+'/crowd_movies/'):

                os.system(f'cp {output_path}syllable_sorted-id-0 (usage)_original-id-5.mp4 /static/img/crowd_movie.mp4')
                return jsonify({'ok': True, 'message': 'Crowd movies successfully created!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-scalar-summary', methods=['GET', 'POST', 'PATCH'])
def viz_plot_scalar_summary(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        index_path = cwd1 + 'moseq2-index.yaml'
        scalars_out = cwd1+'scalars/'

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz plot-scalar-summary --output-file {scalars_out} {index_path}')

            if os.path.exists(cwd1+'/_pca/pca_scores.h5'):
                return jsonify({'ok': True, 'message': 'Scalar values were successfully plotted!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-transition-graph', methods=['GET', 'POST', 'PATCH'])
def viz_plot_trans_graph(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        os.system(f'cd {cwd1}')
        index_path = cwd1 + 'moseq2-index.yaml'
        model_path = cwd1 + 'model.p'

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz plot-transition-graph {index_path} {model_path}')

            if os.path.exists(cwd1+''):
                return jsonify({'ok': True, 'message': 'Transition graphs were successfully plotted!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-usages', methods=['GET', 'POST', 'PATCH'])
def viz_plot_usages(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        os.system(f'cd {cwd1}')

        index_path = cwd1 + 'moseq2-index.yaml'
        model_path = cwd1 + 'model.p'
        usage_path = cwd1 + 'usages/'

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz plot-usages --output-file {usage_path} {index_path} {model_path}')

            if os.path.exists(cwd1+''):
                return jsonify({'ok': True, 'message': 'Syllable usages were successfully plotted!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400