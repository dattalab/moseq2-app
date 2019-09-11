from moseq2_viz.gui import *
import os
from flask import request, jsonify
from app import app, data_path, data_config #, mongo
import logger
import json

@app.route('/viz-add-group', methods=['GET'])
def viz_add_group(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/generate-viz-index', methods=['GET', 'POST', 'PATCH'])
def viz_gen_index():
    cwd = os.getcwd()
    cwd1 = cwd + data_path
    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        index_name = 'moseq2-index.yaml'
        data = request.form.to_dict()
        print(data)
        if data != {}:

            ret = generate_index_command(cwd1, cwd1+'_pca/pca_scores.h5', cwd1+index_name, None, False)

            if os.path.exists(cwd1 + index_name) and ret:
                with open(cwd + data_config + 'sidebar-progress.json') as json_file:
                    data = json.load(json_file)
                    if index_name not in data['model_files']:
                        data['model_files'].append(index_name)

                with open(cwd + data_config + 'sidebar-progress.json', 'w') as outfile:
                    json.dump(data, outfile)
                return jsonify({'ok': True, 'message': 'Index file successfully generated!'}), 200

    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/make-crowd-movies', methods=['GET', 'POST', 'PATCH'])
def viz_make_crowd_movies():
    cwd = os.getcwd()
    cwd1 = cwd + data_path
    if request.method == 'POST':
        # start cli command with default params unless get dict is not empty
        cd_cmd = 'cd ' + cwd1
        os.system(cd_cmd)

        index_name = 'moseq2-index.yaml'
        data = request.form.to_dict()
        print(data)
        if data != {}:
            if os.path.exists(cwd1+'/crowd_movies/'):
                #os.system(f'cp {output_path}syllable_sorted-id-0* {cwd}modules/app/static/img/crowd_movie.mp4')
                return jsonify({'ok': True, 'message': 'Crowd movies successfully created!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/plot-scalar-summary', methods=['GET'])
def viz_plot_scalar_summary(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd+data_path

        index_path = cwd1 + 'moseq2-index.yaml'
        scalars_out = cwd1+'_pca/scalar'

        os.system(f'cd {cwd1}')

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz plot-scalar-summary --output-file {scalars_out} {index_path}')

            if os.path.exists(cwd1+'_pca/scalar_summary.png') and os.path.exists(cwd1+'_pca/scalar_position.png'):
                os.system(f'cp {cwd1}_pca/scalar_summary.png {cwd}/modules/app/static/img/scalar_summary.png')
                os.system(f'cp {cwd1}_pca/scalar_position.png {cwd}/modules/app/static/img/scalar_position.png')
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
            os.system(f'moseq2-viz plot-transition-graph --output-file {cwd1}transition {index_path} {model_path}')

            if os.path.exists(cwd1+'transition.png'):
                os.system(f'cp {cwd1}transition.png {cwd}/modules/app/static/img/transition.png')
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
        usage_path = cwd1 + 'usages'

        if os.path.exists(cwd1):
            os.system(f'moseq2-viz plot-usages --output-file {usage_path} {index_path} {model_path}')

            if os.path.exists(cwd1+'usages.png'):
                os.system(f'cp {cwd1}usages.png {cwd}/modules/app/static/img/usages.png')
                return jsonify({'ok': True, 'message': 'Syllable usages were successfully plotted!'}), 200
            else:
                return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
        else:
            return jsonify({'ok': False, 'message': 'Cannot find input directory!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400