import os
from flask import request, jsonify
from app import app, data_path#, mongo
import logger

@app.route('/learn-model', methods=['GET'])
def model_learn(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd + data_path
        print(cwd1)
        os.system(f'cd {cwd1}')

        os.system(f'moseq2-model learn-model {cwd1}_pca/pca_scores.h5 {cwd1}/model.p')

        if os.path.exists(cwd1 + 'model.p'):
            return jsonify({'ok': True, 'message': 'Model trained.'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/learn-model-parameter-scan', methods=['GET', 'POST', 'PATCH'])
def batch_learn_model(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/aggregate-modeling-results', methods=['GET', 'POST', 'PATCH'])
def batch_agg_modeling_res(path=None):
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400

@app.route('/count-frames', methods=['GET', 'POST', 'PATCH'])
def model_count_frames(path=None):
    if request.method == 'GET':
        cwd = os.getcwd()
        cwd1 = cwd + data_path

        os.system(f'cd {cwd1}')

        read = os.popen(f'moseq2-model count-frames {cwd1}_pca/pca_scores.h5').read()

        if read != None:
            return jsonify({'ok': True, 'message': f'Frames Counted: {read}'}), 200
        else:
            return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400
    return jsonify({'ok': False, 'message': 'Bad request parameters!'}), 400